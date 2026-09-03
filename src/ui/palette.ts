/**
 * Colors follow the dataviz reference palette (validated with
 * scripts/validate_palette.js: all hard gates pass in light mode).
 * Chunk identity uses the categorical slots in fixed order; F uses the slot
 * color, B uses a lighter tint of the same hue so kind is encoded by lightness
 * and chunk by hue. Idle time uses neutral hatching so it never impersonates
 * a series.
 */
export const CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];

export const INK = {
  primary: '#0b0b0b',
  secondary: '#52514e',
  muted: '#898781',
  grid: '#e1e0d9',
  axis: '#c3c2b7',
  surface: '#fcfcfb',
  page: '#f9f9f7',
  sequential: '#3987e5',
  sequentialFill: 'rgba(57, 135, 229, 0.22)',
};

/** Mix a hex color with white by `amount` in [0, 1]. */
export function tint(hex: string, amount: number): string {
  const n = parseInt(hex.slice(1), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  const mix = (c: number) => Math.round(c + (255 - c) * amount);
  return `rgb(${mix(r)}, ${mix(g)}, ${mix(b)})`;
}

export function chunkColor(chunk: number, kind: 'F' | 'B'): string {
  const base = CATEGORICAL[chunk % CATEGORICAL.length];
  return kind === 'F' ? base : tint(base, 0.55);
}
