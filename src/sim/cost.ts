import type { OpKind, SimConfig } from './types.ts';

/**
 * Cost model consulted by the engine. It is deliberately keyed by
 * (kind, mb, chunk, rank) so that v2 can make durations depend on the token
 * count of each micro-batch and on per-chunk layer counts without touching
 * the engine.
 */
export interface CostModel {
  compute(kind: OpKind, mb: number, chunk: number, rank: number): number;
  /** Wire time of one transfer once both peers have arrived. */
  transfer(kind: OpKind, mb: number, from: number, to: number): number;
  /** Input activation of (mb, chunk): allocated when the tensor lands on the rank. */
  activationInput(mb: number, chunk: number, rank: number): number;
  /** Intermediate activations of (mb, chunk): allocated when its forward starts. */
  activationIntermediate(mb: number, chunk: number, rank: number): number;
}

/** Bytes of one chunk input tensor: seq * micro-batch size * hidden * dtype. */
export function inputBytes(cfg: SimConfig): number {
  return (cfg.seqLen ?? 4096) * (cfg.microBatchSize ?? 1) * (cfg.hiddenSize ?? 4096) * (cfg.dtypeBytes ?? 2);
}

/** Transformer layers held by one virtual chunk. */
export function layersPerChunk(cfg: SimConfig): number {
  if (cfg.layersPerChunk !== undefined) return cfg.layersPerChunk;
  return (cfg.numLayers ?? 16) / (cfg.pp * cfg.vpp);
}

/** Total activation retained per (mb, chunk) between forward and backward. */
export function activationBytes(cfg: SimConfig): { input: number; intermediate: number } {
  if (cfg.activationBytes !== undefined) return { input: cfg.activationBytes, intermediate: 0 };
  const input = inputBytes(cfg);
  return { input, intermediate: layersPerChunk(cfg) * (cfg.activationMultiplier ?? 17) * input };
}

/** v1: every micro-batch and chunk costs the same. */
export function constantCost(cfg: SimConfig): CostModel {
  const act = activationBytes(cfg);
  return {
    compute: (kind) => (kind === 'F' ? cfg.forwardTime : cfg.backwardTime),
    transfer: () => cfg.p2pLatency,
    activationInput: () => act.input,
    activationIntermediate: () => act.intermediate,
  };
}
