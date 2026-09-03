import { constantCost } from './cost.ts';
import type { CostModel } from './cost.ts';
import { runProgram } from './engine.ts';
import { computeMemory } from './memory.ts';
import { computeMetrics } from './metrics.ts';
import { buildProgram, SCHEDULES } from './schedules/index.ts';
import type { SimConfig, Trace } from './types.ts';

export * from './types.ts';
export { SCHEDULES, buildProgram } from './schedules/index.ts';
export { runProgram } from './engine.ts';
export { constantCost, inputBytes, activationBytes, layersPerChunk, quadraticShare, computeScale } from './cost.ts';
export type { CostModel } from './cost.ts';

export function validateConfig(cfg: SimConfig): string[] {
  const errors: string[] = [];
  const isInt = (x: number) => Number.isInteger(x);
  if (!isInt(cfg.pp) || cfg.pp < 1) errors.push('pp must be an integer ≥ 1');
  if (!isInt(cfg.vpp) || cfg.vpp < 1) errors.push('vpp must be an integer ≥ 1');
  if (!isInt(cfg.microBatches) || cfg.microBatches < 1) errors.push('micro-batches must be an integer ≥ 1');
  if (!(cfg.forwardTime > 0)) errors.push('forward time must be > 0');
  if (!(cfg.backwardTime > 0)) errors.push('backward time must be > 0');
  if (!(cfg.p2pLatency >= 0)) errors.push('p2p latency must be ≥ 0');
  if (cfg.commModel !== undefined && cfg.commModel !== 'async' && cfg.commModel !== 'sync') errors.push('commModel must be async or sync');
  if (cfg.tokens !== undefined) {
    if (cfg.tokens.length !== cfg.microBatches) errors.push(`tokens has ${cfg.tokens.length} entries, expected ${cfg.microBatches}`);
    if (cfg.tokens.some((x) => !(x > 0))) errors.push('every micro-batch must have > 0 tokens');
  }
  const info = SCHEDULES[cfg.schedule];
  if (!info) errors.push(`unknown schedule: ${cfg.schedule}`);
  else {
    if (!info.supportsVpp && cfg.vpp !== 1) errors.push(`${info.label} requires vpp = 1`);
    if (cfg.schedule === 'interleaved-1f1b' && cfg.microBatches % cfg.pp !== 0) {
      errors.push('Interleaved 1F1B requires micro-batches divisible by pp (as in Megatron)');
    }
  }
  return errors;
}

/** Run the full pipeline: program generation -> DES -> memory -> metrics. */
export function simulate(cfg: SimConfig, cost: CostModel = constantCost(cfg)): Trace {
  const errors = validateConfig(cfg);
  if (errors.length) throw new Error(errors.join('; '));
  const program = buildProgram(cfg);
  const { ops, idles, rankFinish, landed } = runProgram(program, cfg.pp, cost, cfg.commModel ?? 'async');
  const memory = computeMemory(ops, cfg.pp, cost, cfg.baselineBytes ?? 0, landed);
  const metrics = computeMetrics(ops, idles, memory, rankFinish, cfg.pp);
  return { config: cfg, ops, idles, memory, metrics };
}
