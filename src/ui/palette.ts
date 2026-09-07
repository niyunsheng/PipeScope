/**
 * Colors follow the dataviz reference palette (validated with
 * scripts/validate_palette.js: all hard gates pass in light mode).
 * Colour allocation, by role:
 *   - chunk identity: the categorical slots in fixed order (hue);
 *   - F vs B: F is a light tint of the chunk hue, B the full-strength hue, so
 *     kind is lightness and chunk is hue. B stays saturated rather than
 *     shaded so it never drifts towards brown or grey;
 *   - loss: violet, a hue kept out of the chunk slots that matter (purple is
 *     the last categorical slot, only reached at vpp = 8);
 *   - transfers: two light neutral greys on their own rows;
 *   - failure: red, only when something went wrong.
 * No two roles share a hue family, so nothing impersonates anything else.
 */
export const CATEGORICAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#e34948', '#4a3aa7'];

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

/** Forward: light tint of the chunk hue. Backward: the full-strength hue. Loss: violet. */
export function chunkColor(chunk: number, kind: 'F' | 'B' | 'L'): string {
  if (kind === 'L') return LOSS_COLOR;
  const base = CATEGORICAL[chunk % CATEGORICAL.length];
  return kind === 'F' ? tint(base, 0.5) : base;
}

/** Loss ops: violet, away from the chunk hues in use and from the comm greys. */
export const LOSS_COLOR = '#7d3c98';
