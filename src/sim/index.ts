import { constantCost } from './cost.ts';
import type { CostModel } from './cost.ts';
import { runProgram } from './engine.ts';
import { computeMemory } from './memory.ts';
import { computeMetrics } from './metrics.ts';
import { buildProgram, SCHEDULES } from './schedules/index.ts';
import { megatronPlacement } from './defaults.ts';
import { parseFormula } from './formula.ts';
import { warmupVars } from './schedules/custom.ts';
import type { SimConfig, Trace } from './types.ts';

export * from './types.ts';
export { SCHEDULES, buildProgram } from './schedules/index.ts';
export { runProgram } from './engine.ts';
export { constantCost, inputBytes, activationBytes, layersPerChunk, quadraticShare, computeScale } from './cost.ts';
export type { CostModel } from './cost.ts';
export { DEFAULT_CONFIG, builtinWarmupFormula, megatronPlacement } from './defaults.ts';
export { parseFormula, FormulaError } from './formula.ts';
export { warmupVars } from './schedules/custom.ts';

export function validateConfig(cfg: SimConfig): string[] {
  const errors: string[] = [];
  const isInt = (x: number) => Number.isInteger(x);
  if (!isInt(cfg.pp) || cfg.pp < 1) errors.push('pp must be an integer ≥ 1');
  if (!isInt(cfg.vpp) || cfg.vpp < 1) errors.push('vpp must be an integer ≥ 1');
  if (!isInt(cfg.microBatches) || cfg.microBatches < 1) errors.push('micro-batches must be an integer ≥ 1');
  if (!(cfg.forwardTime > 0)) errors.push('forward time must be > 0');
  if (!(cfg.backwardTime > 0)) errors.push('backward time must be > 0');
  if (!(cfg.lossTime >= 0)) errors.push('loss time must be ≥ 0');
  if (!(cfg.p2pLatency >= 0)) errors.push('p2p latency must be ≥ 0');
  if (cfg.commModel !== 'async' && cfg.commModel !== 'sync') errors.push('commModel must be async or sync');
  if (cfg.tokens !== undefined) {
    if (cfg.tokens.length !== cfg.microBatches) errors.push(`tokens has ${cfg.tokens.length} entries, expected ${cfg.microBatches}`);
    if (cfg.tokens.some((x) => !(x > 0))) errors.push('every micro-batch must have > 0 tokens');
  }
  const info = SCHEDULES[cfg.schedule];
  if (!info) errors.push(`unknown schedule: ${cfg.schedule}`);
  else {
    if (!info.supportsVpp && cfg.vpp !== 1) errors.push(`${info.label} requires vpp = 1`);
    if (info.supportsVpp) {
      // Megatron's constraints on the group size (schedules.py L1133-L1153).
      // With the default G = pp this is exactly "m divisible by pp".
      const G = cfg.groupSize;
      if (!isInt(G) || G < cfg.pp || G > cfg.microBatches) errors.push(`group size must be an integer in [pp=${cfg.pp}, m=${cfg.microBatches}]`);
      else if (cfg.microBatches % G !== 0 && cfg.microBatches % G < cfg.pp) errors.push(`m mod G = ${cfg.microBatches % G} must be 0 or ≥ pp (G = pp: m must be divisible by pp)`);
    }
    // With several chunks, only the (send after B, wait before F) placement
    // survives blocking sends: anything else closes the chunk ring (rank pp-1
    // back to rank 0) into a rendezvous cycle. Megatron therefore pairs the
    // overlap path with non-batched isend/irecv, i.e. the async model here.
    if (cfg.vpp > 1 && (cfg.sendAfter !== 'B' || cfg.waitGrad !== 'beforeF') && cfg.commModel !== 'async') {
      errors.push('with vpp > 1 only "send after B" + "wait before F" runs under the sync comm model; use async for the other placements (Megatron does the same: overlap_p2p_comm needs non-batched isend/irecv)');
    }
    // Megatron: the interleaved schedule without p2p overlap needs pp > 2, because
    // with pp = 2 one batched call would carry several sends/recvs between the same
    // two ranks and untagged NCCL p2p cannot pair them (arguments.py L969-L975).
    // Other deadlock-prone combinations are deliberately not pre-judged here: the
    // engine detects them and the timeline shows where the ranks got stuck.
    if (cfg.schedule === 'interleaved-1f1b' && cfg.vpp > 1 && cfg.commModel === 'sync' && cfg.pp <= 2) {
      errors.push('Megatron requires pp > 2 for the interleaved schedule without p2p overlap (sync path): with pp = 2 a batched call would hold several transfers between the same two ranks; use async (overlap path) or pp ≥ 3');
    }
    // Megatron's 1F1B has no non-blocking variant: every p2p call waits. The idealised
    // async version of the same program is available as Custom with vpp = 1.
    if (cfg.schedule === '1f1b' && cfg.commModel !== 'sync') {
      errors.push('1F1B runs under the sync comm model only, as in Megatron; for a non-blocking variant use the Custom schedule with vpp = 1');
    }
    // Built-in schedules only take the placement Megatron implements for them; free combinations live in `custom`.
    if (cfg.schedule === '1f1b' || cfg.schedule === 'interleaved-1f1b') {
      const want = megatronPlacement(cfg.schedule, cfg.commModel);
      if (cfg.sendAfter !== want.sendAfter || cfg.waitGrad !== want.waitGrad) {
        errors.push(`${info.label} under ${cfg.commModel} uses "send after ${want.sendAfter}" + "wait before ${want.waitGrad === 'beforeF' ? 'F' : 'B'}"; use the Custom schedule for other placements`);
      }
    }
    if (cfg.schedule === 'custom') {
      try {
        const f = parseFormula(cfg.warmupFormula);
        const known = Object.keys(warmupVars(cfg, 0));
        const bad = f.vars.filter((v) => !known.includes(v));
        if (bad.length) errors.push(`warmup formula: unknown variable ${bad.join(', ')} (available: ${known.join(', ')})`);
        else for (let r = 0; r < cfg.pp; r++) if (!Number.isFinite(f.eval(warmupVars(cfg, r)))) errors.push(`warmup formula is not finite for r = ${r}`);
      } catch (e) {
        errors.push(`warmup formula: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }
  return errors;
}

/** Run the full pipeline: program generation -> DES -> memory -> metrics. */
export function simulate(cfg: SimConfig, cost: CostModel = constantCost(cfg)): Trace {
  const errors = validateConfig(cfg);
  if (errors.length) throw new Error(errors.join('; '));
  const program = buildProgram(cfg);
  const { ops, idles, rankFinish, transfers, failure } = runProgram(program, cfg.pp, cost, cfg.commModel);
  const memory = computeMemory(ops, cfg.pp, cost, cfg.baselineBytes, transfers);
  const metrics = computeMetrics(ops, idles, memory, rankFinish, cfg.pp);
  return { config: cfg, ops, idles, transfers, memory, metrics, failure };
}
