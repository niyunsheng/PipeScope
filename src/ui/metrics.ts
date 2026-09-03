import { fmt, fmtBytes, pct } from './format.ts';
import { t } from './i18n.ts';
import { chunkColor } from './palette.ts';
import type { Store, UiState } from './state.ts';

/** Stat tiles and the chunk / idle legend. */
export function mountMetrics(root: HTMLElement, store: Store): void {
  const render = (s: UiState): void => {
    root.innerHTML = '';
    if (!s.trace) return;
    const m = s.trace.metrics;

    const tiles = document.createElement('div');
    tiles.className = 'tiles';
    const tile = (label: string, value: string, note = '') => {
      const d = document.createElement('div');
      d.className = 'tile';
      d.innerHTML = `<span class="tile-label">${label}</span><span class="tile-value">${value}</span><span class="tile-note">${note}</span>`;
      tiles.appendChild(d);
    };
    const peak = m.ranks.reduce((a, b) => (b.peakMemory > a.peakMemory ? b : a));
    tile(t('ideal'), fmt(m.idealTime), t('idealNote'));
    tile(t('actual'), fmt(m.totalTime), t('actualNote', { d: fmt(m.totalTime - m.idealTime) }));
    tile(t('bubble'), pct(m.bubbleFraction), t('bubbleNote'));
    tile(t('peak'), fmtBytes(peak.peakMemory), t('peakNote', { r: peak.rank }));
    root.appendChild(tiles);

    // Legend: chunk hue × F/B lightness, plus idle textures.
    const legend = document.createElement('div');
    legend.className = 'legend';
    for (let c = 0; c < s.config.vpp; c++) {
      legend.innerHTML += `<span class="swatch" style="background:${chunkColor(c, 'F')}"></span>${t('legendF', { c })} <span class="swatch" style="background:${chunkColor(c, 'B')}"></span>${t('legendB', { c })} `;
    }
    legend.innerHTML += `<span class="swatch hatch-recv"></span>${t('legendRecv')} <span class="swatch hatch-send"></span>${t('legendSend')}`;
    root.appendChild(legend);

  };
  store.subscribe(render);
}
