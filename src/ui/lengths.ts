/**
 * Micro-batch length generation for heterogeneous-length experiments.
 *
 * Lengths are drawn from a log-normal distribution parameterised by the mean
 * (= seqLen) and the coefficient of variation cv = std / mean, so cv = 0
 * reproduces the homogeneous case. Samples are clamped to [0.1, 4] × mean,
 * then rescaled so the sample mean equals the requested mean exactly: total
 * work is identical across cv values and only the imbalance differs. A seeded
 * PRNG keeps results reproducible and shareable via URL.
 */
export type LengthMode = 'uniform' | 'lognormal' | 'custom';
export type LengthOrder = 'asis' | 'asc' | 'desc' | 'alternate';

export interface LengthSpec {
  n: number;
  mean: number;
  mode: LengthMode;
  cv: number;
  seed: number;
  order: LengthOrder;
  custom?: number[];
}

/** mulberry32: small, fast, seedable PRNG. */
function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(next: () => number): number {
  const u = 1 - next();
  const v = next();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function generateLengths(spec: LengthSpec): number[] {
  let lengths: number[];
  if (spec.mode === 'custom' && spec.custom && spec.custom.length) {
    lengths = Array.from({ length: spec.n }, (_, i) => spec.custom![i % spec.custom!.length]);
  } else if (spec.mode === 'lognormal' && spec.cv > 0) {
    const sigma2 = Math.log(1 + spec.cv * spec.cv);
    const mu = Math.log(spec.mean) - sigma2 / 2;
    const next = rng(spec.seed);
    const raw = Array.from({ length: spec.n }, () => {
      const x = Math.exp(mu + Math.sqrt(sigma2) * gaussian(next));
      return Math.min(4 * spec.mean, Math.max(0.1 * spec.mean, x));
    });
    const sampleMean = raw.reduce((a, b) => a + b, 0) / raw.length;
    lengths = raw.map((x) => Math.max(1, Math.round((x * spec.mean) / sampleMean)));
  } else {
    lengths = Array.from({ length: spec.n }, () => spec.mean);
  }
  return orderLengths(lengths, spec.order);
}

/** Reorder lengths; `alternate` interleaves longest / shortest / 2nd longest / ... */
export function orderLengths(lengths: number[], order: LengthOrder): number[] {
  if (order === 'asis') return lengths.slice();
  const sorted = lengths.slice().sort((a, b) => a - b);
  if (order === 'asc') return sorted;
  if (order === 'desc') return sorted.reverse();
  const out: number[] = [];
  let lo = 0;
  let hi = sorted.length - 1;
  while (lo <= hi) {
    out.push(sorted[hi--]);
    if (lo <= hi) out.push(sorted[lo++]);
  }
  return out;
}

export function parseCustom(text: string): number[] {
  return text
    .split(/[\s,;]+/)
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x) && x > 0)
    .map((x) => Math.round(x));
}

export function summarize(lengths: number[]): { min: number; max: number; mean: number; std: number; cv: number } {
  const n = lengths.length;
  const mean = lengths.reduce((a, b) => a + b, 0) / n;
  const varr = lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(varr);
  return { min: Math.min(...lengths), max: Math.max(...lengths), mean, std, cv: mean > 0 ? std / mean : 0 };
}
