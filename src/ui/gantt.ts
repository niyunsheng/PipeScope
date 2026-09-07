import type { BlockedBy, IdleInterval, Op, TransferRecord } from '../sim/index.ts';
import { fmt, niceStep } from './format.ts';
import { t } from './i18n.ts';
import { INK, chunkColor } from './palette.ts';
import type { Store, UiState } from './state.ts';

export const GUTTER = 64;
/** Failure colour, matching `--critical` in styles.css. */
const FAIL = '#d03b3b';
export const AXIS_H = 24;
/** Height of one compute row; the memory panel uses the same value per rank. */
export const ROW_MAX = 44;

/**
 * Size the canvas backing store for the device pixel ratio and return a
 * cleared 2D context. Assigning `canvas.width` reallocates the bitmap even
 * when the value is unchanged (several MB per frame on a Retina display), so
 * it is only touched when the size actually changes; otherwise the canvas is
 * cleared in place.
 */
export function setupCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number): CanvasRenderingContext2D {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(cssW * dpr);
  const h = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d')!;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
  } else {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, w, h);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

/**
 * Frame-time probe, enabled with `?perf=1`: logs the mean and max draw time
 * of the last 60 frames per panel so slow rendering can be reported.
 */
export function perfProbe(name: string): (ms: number) => void {
  if (!new URLSearchParams(location.search).has('perf')) return () => {};
  const samples: number[] = [];
  return (ms) => {
    samples.push(ms);
    if (samples.length < 60) return;
    const mean = samples.reduce((a, b) => a + b, 0) / samples.length;
    console.info(`[perf] ${name}: mean ${mean.toFixed(2)} ms, max ${Math.max(...samples).toFixed(2)} ms over ${samples.length} frames`);
    samples.length = 0;
  };
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
  transfer?: TransferRecord;
  /** The transfer was hit on the comm lane rather than through a wait. */
  lane?: boolean;
}

/** Follow the primary predecessor from `op` back to the start; returns [op, pred, pred-of-pred, ...]. */
export function criticalChain(ops: Op[], op: Op): Op[] {
  const chain: Op[] = [op];
  const seen = new Set<number>([op.id]);
  let cur = op;
  while (cur.predecessors.length && !seen.has(cur.predecessors[0].op)) {
    cur = ops[cur.predecessors[0].op];
    seen.add(cur.id);
    chain.push(cur);
  }
  return chain;
}

/** Display name of an op kind. */
function kindName(k: Op['kind']): string {
  return k === 'F' ? t('forward') : k === 'B' ? t('backward') : t('loss');
}

function describeDep(b: BlockedBy, ops: Op[], self: Op): string {
  if (b.reason === 'program') return t('depProgram');
  const p = ops[b.op];
  const what = `${kindName(p.kind)} mb${p.mb}`;
  // Same-rank wait: we arrived last and only the wire time of our own transfer remained.
  if (p.rank === self.rank) return t('depWire', { what });
  return b.reason === 'wait-recv' ? t('depWaitRecv', { what, r: p.rank }) : t('depWaitSend', { what, r: p.rank });
}

export function mountGantt(wrapper: HTMLElement, store: Store): void {
  const canvas = document.createElement('canvas');
  canvas.className = 'gantt';
  // The crosshair lives on its own layer so mouse movement never repaints the schedule.
  const overlay = document.createElement('canvas');
  overlay.className = 'overlay';
  const tooltip = document.createElement('div');
  tooltip.className = 'tooltip';
  tooltip.hidden = true;
  wrapper.appendChild(canvas);
  wrapper.appendChild(overlay);
  wrapper.appendChild(tooltip);

  let width = 0;
  let height = 0;
  let rowH = ROW_MAX;
  /** Height of the compute part of a row; equals rowH when the comm lane is off. */
  let compH = ROW_MAX;
  /** Comm-lane bars laid out during the last render, for hit testing. */
  let laneBars: { x0: number; x1: number; y0: number; y1: number; tr: TransferRecord }[] = [];
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
    if (y - AXIS_H - rank * rowH >= compH) {
      // Comm rows (absent when latency is 0, in which case compH === rowH).
      const bar = laneBars.find((b) => b.x0 <= x && x <= b.x1 && b.y0 <= y && y <= b.y1);
      return bar ? { transfer: bar.tr, lane: true } : {};
    }
    const time = xToTime(x, s);
    const op = s.trace.ops.find((o) => o.rank === rank && o.start <= time && time < o.end); // zero-length ops never match
    if (op) return { op };
    const idle = s.trace.idles.find((i) => i.rank === rank && i.start <= time && time < i.end);
    return idle ? { idle, transfer: s.trace.transfers.find((tr) => tr.tag === idle.transferTag) } : {};
  };

  const render = (s: UiState): void => {
    width = wrapper.clientWidth;
    const pp = s.config.pp;
    // Each rank is three sub-rows: compute (ROW_MAX tall), outgoing transfers
    // and incoming transfers (each half of that). Fixed size: the canvas grows
    // with pp and the page scrolls, rather than squeezing rows.
    // With zero latency every bar would be zero-width, so the comm rows are
    // dropped and the timeline is the bare schedule.
    compH = ROW_MAX;
    const laneH = s.config.p2pLatency > 0 ? ROW_MAX : 0;
    rowH = compH + laneH;
    laneBars = [];
    height = AXIS_H + rowH * pp + 4;
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
      const yc = AXIS_H + r * rowH + compH / 2;
      ctx.fillStyle = INK.secondary;
      ctx.fillText(`rank ${r}`, GUTTER - 8, yc);
      if (laneH > 0) {
        ctx.font = '10px system-ui, -apple-system, "Segoe UI", sans-serif';
        ctx.fillStyle = INK.muted;
        ctx.fillText(t('laneOut'), GUTTER - 8, AXIS_H + r * rowH + compH + laneH * 0.25);
        ctx.fillText(t('laneIn'), GUTTER - 8, AXIS_H + r * rowH + compH + laneH * 0.75);
        ctx.font = '12px system-ui, -apple-system, "Segoe UI", sans-serif';
      }
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

    const pad = 4;
    // Only a pinned (clicked) micro-batch is emphasised; hovering shows the
    // tooltip and crosshair alone, so the picture stays still under the mouse.
    const active = s.selectedMb;
    // Transfers that belong to the pinned micro-batch are drawn dark too,
    // so it reads as compute -> wire -> compute.
    const activeTags = new Set(active !== null ? s.trace.transfers.filter((tr) => tr.mb === active).map((tr) => tr.tag) : []);

    // Comm rows: every transfer touching a rank as a bar, outgoing in the
    // first row and incoming in the second. A light segment is the rendezvous
    // wait from posting until data moves; the solid segment is wire time.
    // Overlapping bars stack so simultaneous transfers stay distinguishable.
    // Gaps in the compute row are therefore explained by the rows below it:
    // a rank that is not computing is waiting on one of these bars.
    // The pinned transfer is drawn darker with an outline on both ranks.
    if (laneH > 0) {
      const halfH = laneH / 2;
      const barPad = 1;
      for (let r = 0; r < pp; r++) {
        // Light band behind the two comm rows separates them from compute without lines.
        ctx.fillStyle = INK.page;
        ctx.fillRect(GUTTER, AXIS_H + r * rowH + compH, pw, laneH);
        for (const dir of ['out', 'in'] as const) {
          const bars = s.trace.transfers
            .filter((tr) => (dir === 'out' ? tr.from === r : tr.to === r))
            .map((tr) => ({ tr, posted: dir === 'out' ? tr.sendPosted : tr.recvPosted }))
            .map((b) => ({ ...b, t0: Math.min(b.posted, b.tr.start), t1: b.tr.landed }));
          // Two fixed levels with a fixed meaning: activations (F, flowing
          // downstream) on top, gradients (B, flowing upstream) below. The
          // layout is then identical for every schedule and the level itself
          // tells the kind, which the neutral grey does not.
          const barH = Math.max(2, (halfH - 2 * barPad) / 2);
          const yTop = AXIS_H + r * rowH + compH + (dir === 'out' ? 0 : halfH) + barPad;
          bars.forEach((b) => {
            if (b.t1 < tMin || b.t0 > tMax) return;
            const y0 = yTop + (b.tr.kind === 'F' ? 0 : 1) * barH;
            // Neutral greys: colour is reserved for compute. Direction is the
            // row, kind is in the tooltip; the pinned transfer goes dark.
            const pinnedBar = b.tr.tag === s.selectedTransfer || activeTags.has(b.tr.tag);
            const xs = x(b.tr.start);
            const xe = x(b.tr.landed);
            const xp = x(b.posted);
            // Light greys: wire in muted, rendezvous wait in axis grey (still legible on the row band).
            if (b.posted < b.tr.start - 1e-9) {
              ctx.fillStyle = pinnedBar ? INK.secondary : INK.axis;
              ctx.fillRect(xp, y0, Math.max(1, xs - xp), barH - 1);
            }
            ctx.fillStyle = pinnedBar ? INK.primary : INK.muted;
            ctx.fillRect(xs, y0, Math.max(1.5, xe - xs), barH - 1);
            laneBars.push({ x0: Math.min(xp, xs), x1: Math.max(xe, xs + 1.5), y0, y1: y0 + barH - 1, tr: b.tr });
          });
        }
      }
    }

    // Ops
    ctx.font = `${Math.min(12, compH - 10)}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = 'center';
    const showLabels = compH >= 18;
    for (const op of s.trace.ops) {
      if (op.end < tMin || op.start > tMax) continue;
      if (op.end <= op.start) continue; // zero-length op (loss with lossTime = 0): present, not drawn
      const x0 = x(op.start);
      const w = Math.max(1, x(op.end) - x0 - 1);
      const y = AXIS_H + op.rank * rowH + pad;
      const h = compH - 2 * pad;
      ctx.fillStyle = chunkColor(op.chunk, op.kind);
      ctx.fillRect(x0, y, w, h);
      // No maxWidth argument: fillText with maxWidth takes a slow scaling path.
      if (showLabels && w >= 16) {
        ctx.fillStyle = op.kind === 'F' ? INK.primary : '#ffffff';
        ctx.fillText(op.kind === 'L' ? `L${op.mb}` : String(op.mb), x0 + w / 2, y + h / 2);
      }
    }

    const outline = (op: Op, dashed: boolean) => {
      const x0 = x(op.start);
      const w = Math.max(1, x(op.end) - x0 - 1);
      const y = AXIS_H + op.rank * rowH + pad;
      // White ring then dark outline so the highlight reads on any chunk color.
      ctx.setLineDash([]);
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 4;
      ctx.strokeRect(x0 + 1.5, y + 1.5, Math.max(1, w - 3), compH - 2 * pad - 3);
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.strokeStyle = INK.primary;
      ctx.lineWidth = 2;
      ctx.strokeRect(x0 + 1.5, y + 1.5, Math.max(1, w - 3), compH - 2 * pad - 3);
      ctx.setLineDash([]);
    };
    const center = (op: Op) => ({ cx: (x(op.start) + x(op.end)) / 2, cy: AXIS_H + op.rank * rowH + compH / 2 });
    /** Bar rectangle of `tr` on `rank` (its send bar on the sender, recv bar on the receiver). */
    const barOf = (tr: TransferRecord, rank: number) => {
      const rowTop = AXIS_H + rank * rowH + compH;
      return laneBars.find((b) => b.tr === tr && b.y0 >= rowTop && b.y0 < rowTop + laneH) ?? null;
    };
    /** Straight segment between two op centres (same-rank program order). */
    const link = (a: Op, b: Op, color: string, dashed: boolean) => {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dashed ? [4, 3] : []);
      ctx.beginPath();
      ctx.moveTo(center(a).cx, center(a).cy);
      ctx.lineTo(center(b).cx, center(b).cy);
      ctx.stroke();
      ctx.setLineDash([]);
    };
    /**
     * Show one transfer in context. Its send bar and recv bar are the same
     * event on two rows, so they are joined by a translucent band over their
     * shared interval rather than a line (a line would read as a sequence).
     * Causality is the two short connectors: `a` -> bar on a's rank at the
     * start, and bar on b's rank at the landing -> `b`. `a` and `b` may sit on
     * either end (a wait-send hop runs receiver -> sender).
     */
    const connect = (a: Op | null, tr: TransferRecord, b: Op | null, color = INK.primary, dashed = false) => {
      const barA = a ? barOf(tr, a.rank) : null;
      const barB = b ? barOf(tr, b.rank) : null;
      const send = barOf(tr, tr.from);
      const recv = barOf(tr, tr.to);
      if (!send || !recv) {
        // No comm rows (zero latency) or bars off-screen: fall back to a direct segment.
        if (a && b) link(a, b, color, dashed);
        return;
      }
      const xs = x(tr.start);
      const xe = Math.max(x(tr.landed), xs + 1.5);
      const top = Math.min(send.y0, recv.y0);
      const bottom = Math.max(send.y1, recv.y1);
      ctx.fillStyle = dashed ? 'rgba(11, 11, 11, 0.05)' : 'rgba(11, 11, 11, 0.10)';
      ctx.fillRect(xs, top, xe - xs, bottom - top);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash(dashed ? [4, 3] : []);
      if (a && barA) {
        ctx.beginPath();
        ctx.moveTo(center(a).cx, center(a).cy);
        ctx.lineTo(xs, (barA.y0 + barA.y1) / 2);
        ctx.stroke();
      }
      if (b && barB) {
        ctx.beginPath();
        ctx.moveTo(x(tr.landed), (barB.y0 + barB.y1) / 2);
        ctx.lineTo(center(b).cx, center(b).cy);
        ctx.stroke();
      }
      ctx.setLineDash([]);
    };
    /** The op on the receiving rank that consumes `tr`. */
    const consumerOf = (tr: TransferRecord): Op | null => {
      const prod = tr.producer !== null ? s.trace!.ops[tr.producer] : null;
      if (!prod) return null;
      const stage = tr.kind === 'F' ? prod.stage + 1 : prod.stage - 1;
      return s.trace!.ops.find((o) => o.rank === tr.to && o.mb === tr.mb && o.kind === tr.kind && o.stage === stage) ?? null;
    };
    /** The transfer that delivers `op`'s input, if it came from another stage. */
    const inputOf = (op: Op): TransferRecord | null =>
      s.trace!.transfers.find((tr) => tr.to === op.rank && tr.mb === op.mb && tr.kind === op.kind && tr.producer !== null &&
        s.trace!.ops[tr.producer].stage === (op.kind === 'F' ? op.stage - 1 : op.stage + 1)) ?? null;
    const byTag = (tag: string | undefined) => (tag ? s.trace!.transfers.find((tr) => tr.tag === tag) ?? null : null);

    // Failure marks: the illegal compute step as a red block where it would
    // have run, and every rank stuck in a comm step as a red band from the
    // moment it got stuck to the end of the view.
    if (s.trace.failure) {
      const f = s.trace.failure;
      ctx.fillStyle = 'rgba(208, 59, 59, 0.18)';
      for (const b of f.blocked) {
        ctx.fillRect(x(b.since), AXIS_H + b.rank * rowH + pad, Math.max(2, x(tMax) - x(b.since)), compH - 2 * pad);
      }
      if (f.op) {
        const x0 = x(f.op.start);
        const w = Math.max(6, x(f.op.end) - x0 - 1);
        const y = AXIS_H + f.op.rank * rowH + pad;
        ctx.fillStyle = FAIL;
        ctx.fillRect(x0, y, w, compH - 2 * pad);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(`${f.op.kind}${f.op.mb}!`, x0 + w / 2, y + (compH - 2 * pad) / 2);
      }
    }

    // Critical-predecessor chain of the pinned op: why did it start when it
    // did? Each hop is the single event that released the next op, so the
    // chain is one path back to t = 0 rather than the full dependency DAG.
    const selOp = s.selectedOp !== null ? s.trace.ops[s.selectedOp] : null;
    if (selOp) {
      const chain = criticalChain(s.trace.ops, selOp);
      const inChain = new Set(chain.map((o) => o.id));
      /** One hop op <- pred, drawn through the transfer that linked them when there was one. */
      const hop = (pred: Op, op: Op, b: BlockedBy, color: string, dashed: boolean) => {
        const tr = byTag(b.transferTag);
        if (tr) connect(pred, tr, op, color, dashed);
        else link(pred, op, color, dashed);
      };
      for (let i = 0; i < chain.length; i++) {
        const op = chain[i];
        if (i + 1 < chain.length) hop(chain[i + 1], op, op.predecessors[0], INK.primary, false);
        // Other predecessors that bind at the same instant (ties): drawn as
        // one hop each, not followed further, so the chain stays a line.
        for (const b of op.predecessors.slice(1)) hop(s.trace.ops[b.op], op, b, INK.primary, false);
        // Data input drawn separately (dashed) when it is not what the op waited for.
        const inp = inputOf(op);
        if (inp && inp.producer !== null && !op.predecessors.some((b) => b.op === inp.producer) && !inChain.has(inp.producer)) {
          connect(s.trace.ops[inp.producer], inp, op, INK.secondary, true);
        }
      }
      for (let i = chain.length - 1; i >= 0; i--) outline(chain[i], i !== 0);
    }

    // Path of the active micro-batch across ranks and chunks. Other ops stay
    // fully visible; emphasis comes from a heavy outline plus the connecting
    // line, so the surrounding schedule remains readable as context.
    // Path of the active micro-batch across ranks and chunks, routed through
    // its transfers: compute -> out bar -> in bar -> compute. Other ops stay
    // fully visible; emphasis comes from the heavy outline plus the connector.
    if (active !== null && !selOp) {
      const path = s.trace.ops.filter((o) => o.mb === active).sort((a, b) => a.start - b.start);
      for (let i = 0; i + 1 < path.length; i++) {
        const a = path[i];
        const b = path[i + 1];
        const tr = s.trace.transfers.find((x) => x.producer === a.id && x.to === b.rank && x.kind === b.kind);
        if (tr) connect(a, tr, b);
        else link(a, b, INK.primary, false); // same-rank hop (last stage F -> B)
      }
      for (const op of path) outline(op, false);
    }

    // Pinned transfer: its producer and consumer ops are outlined and joined
    // through the two bars, so the wait it caused can be read in context.
    const pinnedTr = s.selectedTransfer ? s.trace.transfers.find((tr) => tr.tag === s.selectedTransfer) ?? null : null;
    if (pinnedTr) {
      const producer = pinnedTr.producer !== null ? s.trace.ops[pinnedTr.producer] : null;
      const consumer = consumerOf(pinnedTr);
      connect(producer, pinnedTr, consumer);
      if (producer) outline(producer, false);
      if (consumer) outline(consumer, false);
    }
    ctx.restore();
  };

  const renderOverlay = (s: UiState): void => {
    const ctx = setupCanvas(overlay, width, height);
    if (s.trace) drawCrosshair(ctx, s, rowH * s.config.pp);
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
      if (s.hoverTime !== null) store.set({ hoverTime: null });
      return;
    }
    const hit = hitTest(mx, my, s);
    // Named `time`, not `t`: `t` is the i18n function used below.
    const time = xToTime(mx, s);
    if (hit.op) {
      const o = hit.op;
      const tok = s.config.tokens ? ` · ${s.config.tokens[o.mb]} ${t('tokens')}` : '';
      const dep = o.predecessors.length ? `<br>${t('startedAfter')}: ${o.predecessors.map((b) => describeDep(b, s.trace.ops, o)).join(' · ')}` : '';
      tooltip.innerHTML = `<b>${kindName(o.kind)} · ${t('microBatch')} ${o.mb}</b>${tok}<br>${t('opWhere', { r: o.rank, c: o.chunk, s: o.stage })}<br>${fmt(o.start)} – ${fmt(o.end)} (${fmt(o.end - o.start)})${dep}`;
    } else if (hit.lane && hit.transfer) {
      const tr = hit.transfer;
      const wait = Math.max(0, tr.start - Math.min(tr.sendPosted, tr.recvPosted));
      tooltip.innerHTML = `<b>${t('transferTitle', { kind: tr.kind === 'F' ? t('forward') : t('backward'), mb: tr.mb, a: tr.from, b: tr.to })}</b><br>${t('transferTimes', { s: fmt(tr.sendPosted), r: fmt(tr.recvPosted), d0: fmt(tr.start), d1: fmt(tr.landed) })}<br>${t('transferWait')} ${fmt(wait)} · ${t('transferWire')} ${fmt(tr.landed - tr.start)}`;
    } else if (hit.idle) {
      const i = hit.idle;
      const tr = hit.transfer;
      const what = tr ? `<br>${t('transferTitle', { kind: tr.kind === 'F' ? t('forward') : t('backward'), mb: tr.mb, a: tr.from, b: tr.to })}<br>${t('transferTimes', { s: fmt(tr.sendPosted), r: fmt(tr.recvPosted), d0: fmt(tr.start), d1: fmt(tr.landed) })}` : '';
      tooltip.innerHTML = `<b>${i.reason === 'wait-recv' ? t('waitingRecv') : t('waitingSend')}</b> · rank ${i.rank}${what}<br>${fmt(i.start)} – ${fmt(i.end)} · ${t('waitDetail', { d: fmt(i.end - i.start), a: fmt(i.peerWait), b: fmt(i.transfer) })}`;
    } else {
      tooltip.innerHTML = `t = ${fmt(time)}`;
    }
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(mx + 12, width - 240)}px`;
    tooltip.style.top = `${my + 12}px`;
    store.set({ hoverTime: time });
  });
  canvas.addEventListener('mouseleave', () => {
    tooltip.hidden = true;
    dragging = null;
    store.set({ hoverTime: null });
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
    if (e.shiftKey) {
      // Shift-click pins a single op and shows its critical-predecessor chain.
      const id = hit.op ? hit.op.id : null;
      store.set({ selectedOp: id !== null && s.selectedOp === id ? null : id, selectedMb: null, selectedTransfer: null });
      return;
    }
    if (hit.idle || hit.lane) {
      // Clicking a wait (or a lane bar) pins the transfer and frames both ends.
      const tag = hit.transfer ? hit.transfer.tag : null;
      store.set({ selectedTransfer: tag !== null && s.selectedTransfer === tag ? null : tag, selectedMb: null, selectedOp: null });
      return;
    }
    const mb = hit.op ? hit.op.mb : null;
    store.set({ selectedMb: mb !== null && s.selectedMb === mb ? null : mb, selectedOp: null, selectedTransfer: null });
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

  // Coalesce redraws to one per animation frame: several store updates can
  // land between frames (hover + crosshair), and drawing twice is wasted.
  // Two layers, two cadences: the schedule repaints only when something it
  // shows changed; the crosshair overlay repaints on every hover move.
  const STATIC_KEYS: (keyof UiState)[] = ['trace', 'scale', 'config', 'selectedMb', 'selectedOp', 'selectedTransfer'];
  let prev: UiState | null = null;
  let frame = 0;
  let needStatic = true;
  const probe = perfProbe('gantt');
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
    const s = store.get();
    if (s.trace && s.scale.offset === 0) fit(s);
    else {
      render(s);
      renderOverlay(s);
    }
  }).observe(wrapper);
}
