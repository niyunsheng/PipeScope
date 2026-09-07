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
  return cfg.seqLen * cfg.microBatchSize * cfg.hiddenSize * cfg.dtypeBytes;
}

/** Transformer layers held by one virtual chunk (stage). */
export function layersPerChunk(cfg: SimConfig): number {
  return cfg.numLayers / (cfg.pp * cfg.vpp);
}

/** Total activation retained per (mb, chunk) between forward and backward. */
export function activationBytes(cfg: SimConfig): { input: number; intermediate: number } {
  if (cfg.activationBytes !== undefined) return { input: cfg.activationBytes, intermediate: 0 };
  const input = inputBytes(cfg);
  return { input, intermediate: layersPerChunk(cfg) * cfg.activationMultiplier * input };
}

/**
 * Quadratic (core-attention) share of per-layer compute at the reference
 * length: s / (k * h + s), with k = linear / attention FLOP coefficient ratio.
 */
export function quadraticShare(cfg: Pick<SimConfig, 'seqLen' | 'hiddenSize' | 'linearAttnRatio'>): number {
  const s = cfg.seqLen;
  const h = cfg.hiddenSize;
  const k = cfg.linearAttnRatio;
  return s / (k * h + s);
}

/** Relative compute cost of a micro-batch with `ratio` = tokens / seqLen. */
export function computeScale(ratio: number, alpha: number): number {
  return (1 - alpha) * ratio + alpha * ratio * ratio;
}

/**
 * Default cost model. Per-layer times × layers per chunk; micro-batches scale with
 * their token count when `cfg.tokens` is given (v2), otherwise all are equal.
 */
export function constantCost(cfg: SimConfig): CostModel {
  const act = activationBytes(cfg);
  const layers = layersPerChunk(cfg);
  const L0 = cfg.seqLen;
  const alpha = quadraticShare(cfg);
  const ratio = (mb: number) => (cfg.tokens && cfg.tokens[mb] !== undefined ? cfg.tokens[mb] / L0 : 1);
  return {
    // Loss is per token (cross-entropy), so it scales linearly with length.
    compute: (kind, mb) =>
      kind === 'L' ? cfg.lossTime * ratio(mb) : layers * (kind === 'F' ? cfg.forwardTime : cfg.backwardTime) * computeScale(ratio(mb), alpha),
    transfer: () => cfg.p2pLatency,
    activationInput: (mb) => act.input * ratio(mb),
    activationIntermediate: (mb) => act.intermediate * ratio(mb),
  };
}
