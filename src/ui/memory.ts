import { fmt, fmtBytes } from './format.ts';
import { t } from './i18n.ts';
import { AXIS_H, GUTTER, ROW_MAX, drawAxis, drawCrosshair, perfProbe, setupCanvas } from './gantt.ts';
import { INK } from './palette.ts';
import type { Store, UiState } from './state.ts';

/**
 * Memory panel: one small-multiple row per rank, aligned with the Gantt rows
 * and sharing its time scale. A single sequential hue is used because the
 * rows are already identified by position, so no legend is required.
 */
export function mountMemory(wrapper: HTMLElement, store: Store): void {
  const canvas = document.createElement('canvas');
  canvas.className = 'memory';
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  tooltip.hidden = true;
  const overlay = document.createElement('canvas');
  overlay.className = 'overlay';
  wrapper.appendChild(canvas);
  wrapper.appendChild(overlay);
  wrapper.appendChild(tooltip);

  let width = 0;
  let height = 0;

  const render = (s: UiState): void => {
    width = wrapper.clientWidth;
    const pp = s.config.pp;
    const rowH = ROW_MAX;
    height = AXIS_H + rowH * pp + 4;
    const ctx = setupCanvas(canvas, width, height);
    ctx.fillStyle = INK.surface;
    ctx.fillRect(0, 0, width, height);
    if (!s.trace) return;
    const pw = Math.max(1, width - GUTTER - 8);
    const { pxPerUnit, offset } = s.scale;
    const x = (t: number) => GUTTER + (t - offset) * pxPerUnit;
    const maxBytes = Math.max(1e-9, ...s.trace.metrics.ranks.map((r) => r.peakMemory));
    const total = s.trace.metrics.totalTime;

    drawAxis(ctx, s, pw, rowH * pp);
    ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, AXIS_H, pw, rowH * pp);
    ctx.clip();
    for (let r = 0; r < pp; r++) {
      const yBase = AXIS_H + (r + 1) * rowH - 3;
      const yScale = (rowH - 8) / maxBytes;
      const samples = s.trace.memory[r];
      // Step area
      ctx.beginPath();
      ctx.moveTo(x(0), yBase);
      let prev = samples[0].bytes;
      for (const smp of samples) {
        ctx.lineTo(x(smp.t), yBase - prev * yScale);
        ctx.lineTo(x(smp.t), yBase - smp.bytes * yScale);
        prev = smp.bytes;
      }
      ctx.lineTo(x(total), yBase - prev * yScale);
      ctx.lineTo(x(total), yBase);
      ctx.closePath();
      ctx.fillStyle = INK.sequentialFill;
      ctx.fill();
      ctx.strokeStyle = INK.sequential;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      // Row separator
      ctx.strokeStyle = INK.grid;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(GUTTER, AXIS_H + (r + 1) * rowH + 0.5);
      ctx.lineTo(GUTTER + pw, AXIS_H + (r + 1) * rowH + 0.5);
      ctx.stroke();
    }
    ctx.restore();
    // Row labels with peak
    ctx.textAlign = 'right';
    for (let r = 0; r < pp; r++) {
      const yc = AXIS_H + r * rowH + rowH / 2;
      ctx.fillStyle = INK.secondary;
      ctx.fillText(`rank ${r}`, GUTTER - 8, yc - 7);
      ctx.fillStyle = INK.muted;
      ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.fillText(fmtBytes(s.trace.metrics.ranks[r].peakMemory), GUTTER - 8, yc + 7);
      ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
    }
  };

  /** Crosshair and the per-rank value under it; repainted on every hover move. */
  const renderOverlay = (s: UiState): void => {
    const ctx = setupCanvas(overlay, width, height);
    if (!s.trace || s.hoverTime === null) return;
    const pp = s.config.pp;
    const rowH = ROW_MAX;
    const pw = Math.max(1, width - GUTTER - 8);
    const x = (t: number) => GUTTER + (t - s.scale.offset) * s.scale.pxPerUnit;
    ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.fillStyle = INK.primary;
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, AXIS_H, pw, rowH * pp);
    ctx.clip();
    for (let r = 0; r < pp; r++) {
      const samples = s.trace.memory[r];
      let cur = samples[0];
      for (const smp of samples) if (smp.t <= s.hoverTime) cur = smp;
      ctx.fillText(fmtBytes(cur.bytes), x(s.hoverTime) + 6, AXIS_H + r * rowH + 10);
    }
    ctx.restore();
    drawCrosshair(ctx, s, rowH * pp);
  };

  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const s = store.get();
    if (!s.trace || mx < GUTTER) {
      tooltip.hidden = true;
      return;
    }
    const time = s.scale.offset + (mx - GUTTER) / s.scale.pxPerUnit;
    const pp = s.config.pp;
    const rowH = ROW_MAX;
    const rank = Math.floor((my - AXIS_H) / rowH);
    if (rank >= 0 && rank < pp) {
      let cur = s.trace.memory[rank][0];
      for (const smp of s.trace.memory[rank]) if (smp.t <= time) cur = smp;
      const resident = cur.resident.length ? cur.resident.map((k) => `mb${k.split(':')[0]}·c${k.split(':')[1]}`).join(', ') : t('none');
      tooltip.innerHTML = `<b>rank ${rank} · t = ${fmt(time)}</b><br>${t('memory')} ${fmtBytes(cur.bytes)}<br>${t('resident')}: ${resident}`;
      tooltip.hidden = false;
      tooltip.style.left = `${Math.min(mx + 12, wrapper.clientWidth - 260)}px`;
      tooltip.style.top = `${my + 12}px`;
    } else {
      tooltip.hidden = true;
    }
    store.set({ hoverTime: time });
  });
  canvas.addEventListener('mouseleave', () => {
    tooltip.hidden = true;
    store.set({ hoverTime: null });
  });

  // Coalesce redraws to one per animation frame: several store updates can
  // land between frames (hover + crosshair), and drawing twice is wasted.
  const STATIC_KEYS: (keyof UiState)[] = ['trace', 'scale', 'config'];
  let prev: UiState | null = null;
  let frame = 0;
  let needStatic = true;
  const probe = perfProbe('memory');
  store.subscribe((s) => {
    if (!prev || STATIC_KEYS.some((k) => prev![k] !== s[k])) needStatic = true;
    prev = s;
    if (frame) return;
    frame = requestAnimationFrame(() => {
      frame = 0;
      const t0 = performance.now();
      const cur = store.get();
      if (needStatic) {
        needStatic = false;
        render(cur);
      }
      renderOverlay(cur);
      probe(performance.now() - t0);
    });
  });
  new ResizeObserver(() => {
    render(store.get());
    renderOverlay(store.get());
  }).observe(wrapper);
}
