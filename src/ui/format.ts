export function fmt(x: number, digits = 2): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(digits);
}

/** Human-readable byte size using 1024-based units (GPU memory convention). */
export function fmtBytes(bytes: number): string {
  const abs = Math.abs(bytes);
  if (abs < 1024) return `${fmt(bytes, 0)} B`;
  if (abs < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (abs < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

export function pct(x: number, digits = 2): string {
  return `${(x * 100).toFixed(digits)}%`;
}

/** Choose a "nice" tick step so that roughly `target` ticks fit in `range`. */
export function niceStep(range: number, target: number): number {
  const raw = range / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (raw <= m * mag) return m * mag;
  return 10 * mag;
}
