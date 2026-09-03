import type { IdleInterval, Op } from '../sim/index.ts';
import { fmt, niceStep } from './format.ts';
import { t } from './i18n.ts';
import { INK, chunkColor } from './palette.ts';
import { activeMb } from './state.ts';
import type { Store, UiState } from './state.ts';

export const GUTTER = 64;
export const AXIS_H = 24;
export const ROW_MIN = 22;
export const ROW_MAX = 44;

export function rowHeight(pp: number, available: number): number {
  return Math.max(ROW_MIN, Math.min(ROW_MAX, Math.floor(available / Math.max(1, pp))));
}

/** Set canvas backing store for the device pixel ratio; returns the 2D context. */
export function setupCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

let hatchCache: Record<string, CanvasPattern | null> = {};

/** Diagonal hatch pattern; 45° for recv waits, 135° for send waits (texture, not hue, carries the distinction). */
export function hatch(ctx: CanvasRenderingContext2D, dir: 'recv' | 'send'): CanvasPattern | null {
  if (hatchCache[dir]) return hatchCache[dir];
  const c = document.createElement('canvas');
  c.width = 8;
  c.height = 8;
  const g = c.getContext('2d')!;
  g.strokeStyle = INK.axis;
  g.lineWidth = 1.2;
  g.beginPath();
  if (dir === 'recv') {
    g.moveTo(0, 8);
    g.lineTo(8, 0);
  } else {
    g.moveTo(0, 0);
    g.lineTo(8, 8);
  }
  g.stroke();
  const p = ctx.createPattern(c, 'repeat');
  hatchCache[dir] = p;
  return p;
}

/** Draw the shared time axis at the top of a panel. */
export function drawAxis(ctx: CanvasRenderingContext2D, s: UiState, plotW: number, plotH: number): void {
  const { pxPerUnit, offset } = s.scale;
  const tMin = offset;
  const tMax = offset + plotW / pxPerUnit;
  const step = niceStep(tMax - tMin, plotW / 80);
  ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'center';
  for (let t = Math.ceil(tMin / step) * step; t <= tMax; t += step) {
    const x = GUTTER + (t - offset) * pxPerUnit;
    ctx.strokeStyle = INK.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(Math.round(x) + 0.5, AXIS_H);
    ctx.lineTo(Math.round(x) + 0.5, AXIS_H + plotH);
    ctx.stroke();
    ctx.fillStyle = INK.muted;
    ctx.fillText(fmt(Number(t.toFixed(6))), x, AXIS_H / 2);
  }
  ctx.strokeStyle = INK.axis;
  ctx.beginPath();
  ctx.moveTo(GUTTER, AXIS_H + 0.5);
  ctx.lineTo(GUTTER + plotW, AXIS_H + 0.5);
  ctx.stroke();
}

export function drawCrosshair(ctx: CanvasRenderingContext2D, s: UiState, plotH: number): void {
  if (s.hoverTime === null) return;
  const x = GUTTER + (s.hoverTime - s.scale.offset) * s.scale.pxPerUnit;
  ctx.strokeStyle = INK.secondary;
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(Math.round(x) + 0.5, AXIS_H);
  ctx.lineTo(Math.round(x) + 0.5, AXIS_H + plotH);
  ctx.stroke();
  ctx.setLineDash([]);
}

interface Hit {
  op?: Op;
  idle?: IdleInterval;
}

export function mountGantt(wrapper: HTMLElement, store: Store): void {
  const canvas = document.createElement('canvas');
  canvas.className = 'gantt';
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  tooltip.hidden = true;
  wrapper.appendChild(canvas);
  wrapper.appendChild(tooltip);

  let width = 0;
  let rowH = ROW_MAX;
  let dragging: { x: number; offset: number } | null = null;
  let lastTraceId: object | null = null;

  const plotW = () => Math.max(1, width - GUTTER - 8);

  const fit = (s: UiState): void => {
    if (!s.trace) return;
    const total = Math.max(1e-9, s.trace.metrics.totalTime);
    store.set({ scale: { pxPerUnit: plotW() / total, offset: 0 } });
  };

  const xToTime = (x: number, s: UiState) => s.scale.offset + (x - GUTTER) / s.scale.pxPerUnit;

  const hitTest = (x: number, y: number, s: UiState): Hit => {
    if (!s.trace || x < GUTTER || y < AXIS_H) return {};
    const rank = Math.floor((y - AXIS_H) / rowH);
    if (rank < 0 || rank >= s.config.pp) return {};
    const t = xToTime(x, s);
    const op = s.trace.ops.find((o) => o.rank === rank && o.start <= t && t < o.end);
    if (op) return { op };
    const idle = s.trace.idles.find((i) => i.rank === rank && i.start <= t && t < i.end);
    return { idle };
  };

  const render = (s: UiState): void => {
    width = wrapper.clientWidth;
    const pp = s.config.pp;
    rowH = rowHeight(pp, Math.min(ROW_MAX * pp, wrapper.clientHeight - AXIS_H || ROW_MAX * pp));
    const height = AXIS_H + rowH * pp + 4;
    const ctx = setupCanvas(canvas, width, height);
    ctx.fillStyle = INK.surface;
    ctx.fillRect(0, 0, width, height);
    if (!s.trace) return;
    if (s.trace !== lastTraceId) {
      lastTraceId = s.trace;
      fit(s);
      return; // fit() triggers another render through the store
    }
    const { pxPerUnit, offset } = s.scale;
    const pw = plotW();
    const tMin = offset;
    const tMax = offset + pw / pxPerUnit;
    const x = (t: number) => GUTTER + (t - offset) * pxPerUnit;

    drawAxis(ctx, s, pw, rowH * pp);

    // Row labels
    ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let r = 0; r < pp; r++) {
      const yc = AXIS_H + r * rowH + rowH / 2;
      ctx.fillStyle = INK.secondary;
      ctx.fillText(`rank ${r}`, GUTTER - 8, yc);
      ctx.strokeStyle = INK.grid;
      ctx.beginPath();
      ctx.moveTo(GUTTER, AXIS_H + (r + 1) * rowH + 0.5);
      ctx.lineTo(GUTTER + pw, AXIS_H + (r + 1) * rowH + 0.5);
      ctx.stroke();
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(GUTTER, AXIS_H, pw, rowH * pp);
    ctx.clip();

    // Idle intervals as hatching
    const pad = 4;
    for (const idle of s.trace.idles) {
      if (idle.end < tMin || idle.start > tMax) continue;
      const y = AXIS_H + idle.rank * rowH + pad;
      const p = hatch(ctx, idle.reason === 'wait-recv' ? 'recv' : 'send');
      if (!p) continue;
      ctx.fillStyle = p;
      ctx.fillRect(x(idle.start), y, Math.max(1, x(idle.end) - x(idle.start)), rowH - 2 * pad);
    }

    // Ops
    const active = activeMb(s);
    ctx.font = `${Math.min(12, rowH - 10)}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = 'center';
    const showLabels = rowH >= 18;
    for (const op of s.trace.ops) {
      if (op.end < tMin || op.start > tMax) continue;
      const x0 = x(op.start);
      const w = Math.max(1, x(op.end) - x0 - 1);
      const y = AXIS_H + op.rank * rowH + pad;
      const h = rowH - 2 * pad;
      ctx.fillStyle = chunkColor(op.chunk, op.kind);
      ctx.fillRect(x0, y, w, h);
      if (showLabels && w >= 16) {
        ctx.fillStyle = op.kind === 'F' ? '#ffffff' : INK.primary;
        ctx.fillText(String(op.mb), x0 + w / 2, y + h / 2, w - 2);
      }
    }

    // Path of the active micro-batch across ranks and chunks. Other ops stay
    // fully visible; emphasis comes from a heavy outline plus the connecting
    // line, so the surrounding schedule remains readable as context.
    if (active !== null) {
      const path = s.trace.ops.filter((o) => o.mb === active).sort((a, b) => a.start - b.start);
      ctx.strokeStyle = INK.primary;
      ctx.lineWidth = 2;
      ctx.beginPath();
      path.forEach((op, i) => {
        const cx = (x(op.start) + x(op.end)) / 2;
        const cy = AXIS_H + op.rank * rowH + rowH / 2;
        if (i === 0) ctx.moveTo(cx, cy);
        else ctx.lineTo(cx, cy);
      });
      ctx.stroke();
      for (const op of path) {
        const x0 = x(op.start);
        const w = Math.max(1, x(op.end) - x0 - 1);
        const y = AXIS_H + op.rank * rowH + pad;
        // White ring then dark outline so the highlight reads on any chunk color.
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 4;
        ctx.strokeRect(x0 + 1.5, y + 1.5, Math.max(1, w - 3), rowH - 2 * pad - 3);
        ctx.strokeStyle = INK.primary;
        ctx.lineWidth = 2;
        ctx.strokeRect(x0 + 1.5, y + 1.5, Math.max(1, w - 3), rowH - 2 * pad - 3);
      }
    }
    ctx.restore();
    drawCrosshair(ctx, s, rowH * pp);
  };

  // --- interaction ---
  canvas.addEventListener('mousemove', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const s = store.get();
    if (dragging) {
      const dx = mx - dragging.x;
      store.set({ scale: { ...s.scale, offset: dragging.offset - dx / s.scale.pxPerUnit } });
      return;
    }
    if (!s.trace || mx < GUTTER) {
      tooltip.hidden = true;
      if (s.hoverMb !== null || s.hoverTime !== null) store.set({ hoverMb: null, hoverTime: null });
      return;
    }
    const hit = hitTest(mx, my, s);
    const t = xToTime(mx, s);
    const hoverMb = hit.op ? hit.op.mb : null;
    if (hit.op) {
      const o = hit.op;
      tooltip.innerHTML = `<b>${o.kind === 'F' ? t('forward') : t('backward')} · ${t('microBatch')} ${o.mb}</b><br>${t('opWhere', { r: o.rank, c: o.chunk, s: o.stage })}<br>${fmt(o.start)} – ${fmt(o.end)} (${fmt(o.end - o.start)})`;
    } else if (hit.idle) {
      const i = hit.idle;
      tooltip.innerHTML = `<b>${i.reason === 'wait-recv' ? t('waitingRecv') : t('waitingSend')}</b> · rank ${i.rank}<br>${i.detail}<br>${fmt(i.start)} – ${fmt(i.end)} · ${t('waitDetail', { a: fmt(i.peerWait), b: fmt(i.transfer) })}`;
    } else {
      tooltip.innerHTML = `t = ${fmt(t)}`;
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(mx + 12, width - 240)}px`;
    tooltip.style.top = `${my + 12}px`;
    store.set({ hoverMb, hoverTime: t });
  });
  canvas.addEventListener('mouseleave', () => {
    tooltip.hidden = true;
    dragging = null;
    store.set({ hoverMb: null, hoverTime: null });
  });
  canvas.addEventListener('mousedown', (e) => {
    const rect = canvas.getBoundingClientRect();
    dragging = { x: e.clientX - rect.left, offset: store.get().scale.offset };
  });
  canvas.addEventListener('mouseup', (e) => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const moved = dragging ? Math.abs(mx - dragging.x) > 3 : false;
    dragging = null;
    if (moved) return;
    const s = store.get();
    const hit = hitTest(mx, e.clientY - rect.top, s);
    const mb = hit.op ? hit.op.mb : null;
    store.set({ selectedMb: mb !== null && s.selectedMb === mb ? null : mb });
  });
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const s = store.get();
      if (!s.trace) return;
      const factor = Math.exp(-e.deltaY * 0.0015);
      const minPx = plotW() / Math.max(1e-9, s.trace.metrics.totalTime);
      const px = Math.max(minPx, s.scale.pxPerUnit * factor);
      const tAtCursor = xToTime(mx, s);
      let offset = tAtCursor - (mx - GUTTER) / px;
      offset = Math.max(0, Math.min(offset, s.trace.metrics.totalTime - plotW() / px));
      store.set({ scale: { pxPerUnit: px, offset } });
    },
    { passive: false },
  );
  canvas.addEventListener('dblclick', () => fit(store.get()));

  store.subscribe(render);
  new ResizeObserver(() => {
    const s = store.get();
    if (s.trace && s.scale.offset === 0) fit(s);
    else render(s);
  }).observe(wrapper);
}
