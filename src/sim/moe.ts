import type { ComputeStep, Op, Program, SimConfig, Step } from './types.ts';
import type { CostModel } from './cost.ts';
import { layersPerChunk } from './cost.ts';

export const parseMoeRatios = (value: string): number[] => value.trim().split(/[\s,/:]+/).map(Number);

/** Combine only explicitly marked steady pairs. Both inputs must be ready before
 * entering Megatron's fixed layer schedule; never search for another partner.
 * Source: Megatron-LM 630956b357d4, TransformerLayerSchedulePlan.run.
 */
export function combineMoe(program: Program): Program {
  return program.map(steps => {
    const result: Step[] = [];
    for (let i = 0; i < steps.length; i++) {
      const f = steps[i];
      if (f.type !== 'compute' || !f.steady) { result.push(f); continue; }
      const before: Step[] = [], after: Step[] = [];
      let b: ComputeStep | undefined;
      while (++i < steps.length) {
        const s = steps[i];
        if (s.type === 'compute') { b = s; break; }
        if (s.type === 'wait') before.push(s);
        else if (s.type === 'comm') {
          if (s.recvs.length) before.push({ ...s, sends: [] });
          if (s.sends.length) after.push({ ...s, recvs: [] });
        } else after.push(s);
      }
      if (!b || b.kind !== 'B') throw new Error('MoE overlap: missing fixed backward partner');
      result.push(...before, { ...f, backward: { mb: b.mb, chunk: b.chunk } }, ...after);
    }
    return result;
  });
}

/** Deterministic two-stream event schedule. Each issued node waits for the last
 * event of its own pass and the previous node on its stream. No greedy reorder.
 * EP ranks are represented by one balanced collective cost per PP rank.
 */
export function moeTiming(f: ComputeStep, rank: number, start: number, cfg: SimConfig, cost: CostModel) {
  const ratios = parseMoeRatios(cfg.moeRatios).map(x => x / 100);
  const layers = layersPerChunk(cfg);
  const segments: [NonNullable<Op['segments']>, NonNullable<Op['segments']>] = [[], []];
  const ready = [start, start];
  const streams = { compute: start, ep: start };
  const passes = [f, f.backward ? { ...f.backward, kind: 'B' as const } : null];
  const issue = (pass: number, part: number, layer: number) => {
    const op = passes[pass];
    if (!op) return;
    const resource = part === 0 || part === 2 ? 'compute' : 'ep';
    const begin = Math.max(ready[pass], streams[resource]);
    const end = begin + cost.compute(op.kind, op.mb, op.chunk, rank) / layers * ratios[part];
    segments[pass].push({ part: ['attn', 'dispatch', 'experts', 'combine'][part], layer, start: begin, end, resource });
    ready[pass] = streams[resource] = end;
  };
  for (let layer = 0; layer < layers; layer++) {
    if (f.backward) {
      const bl = layers - 1 - layer;
      issue(1, 3, bl); issue(0, 0, layer); issue(1, 2, bl); issue(0, 1, layer);
      issue(1, 1, bl); issue(0, 2, layer); issue(0, 3, layer); issue(1, 0, bl);
    } else {
      for (const part of f.kind === 'B' ? [3, 2, 1, 0] : [0, 1, 2, 3]) issue(0, part, f.kind === 'B' ? layers - 1 - layer : layer);
    }
  }
  return { segments, end: Math.max(...ready) };
}
