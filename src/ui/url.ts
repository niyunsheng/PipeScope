import { SCHEDULES, megatronPlacement } from '../sim/index.ts';
import type { SimConfig } from '../sim/index.ts';

/**
 * Shareable URLs. Only settings that differ from the defaults are written,
 * one `key=value` pair per field named exactly as in `SimConfig`, so the
 * default view is the bare page and a typical link looks like
 * `?schedule=1f1b&p2pLatency=0.5&sel=3`. `tokens` is a comma list.
 * The query is assembled by hand so commas stay readable; only values with
 * characters that are unsafe in a query (spaces, `+`, `&`, ...) are escaped,
 * which in practice is just the warmup formula.
 */
/**
 * Parse the query by hand: URLSearchParams turns `+` into a space, but `+` is
 * an operator in the warmup formula and we want it to stay literal.
 */
function parseQuery(search: string): Map<string, string> {
  const q = new Map<string, string>();
  for (const part of search.replace(/^\?/, '').split('&')) {
    if (!part) continue;
    const i = part.indexOf('=');
    const k = decodeURIComponent(i < 0 ? part : part.slice(0, i));
    const v = i < 0 ? '' : decodeURIComponent(part.slice(i + 1));
    q.set(k, v);
  }
  return q;
}

/** Escape only what would break the query itself; `+ ( ) *` stay readable. */
const enc = (v: string) => v.replace(/[%&#=?\s]/g, (c) => (/\s/.test(c) ? '' : encodeURIComponent(c)));

export function readUrl(defaults: SimConfig): { config: SimConfig; selectedMb: number | null } {
  const q = parseQuery(location.search);
  const config = { ...defaults } as Record<string, unknown>;
  for (const [key, def] of Object.entries(defaults)) {
    const raw = q.get(key);
    if (raw === undefined) continue;
    if (typeof def === 'number') {
      const v = Number(raw);
      if (Number.isFinite(v)) config[key] = v;
    } else if (typeof def === 'boolean') {
      config[key] = raw === 'true' || raw === '1';
    } else {
      config[key] = raw;
    }
  }
  const toks = q.get('tokens');
  if (toks) config.tokens = toks.split(',').map(Number).filter((x) => x > 0);
  // Group size defaults to pp; a link that changes pp but not G means G = pp.
  if (!q.has('groupSize') && q.has('pp')) config.groupSize = config.pp;
  // Controls that are greyed out in the UI ignore the link: blocking-only
  // schedules are always `sync`, the built-in schedules always use Megatron's
  // placement, and warmup / cooldown prefetch exists only on the interleaved
  // skeleton's isend / irecv path.
  const schedule = config.schedule as SimConfig['schedule'];
  if (schedule === '1f1b' || schedule === 'gpipe') config.commModel = 'sync';
  if (schedule !== 'custom') Object.assign(config, megatronPlacement(schedule, config.commModel as SimConfig['commModel']));
  if (!((schedule === 'interleaved-1f1b' || schedule === 'custom') && config.commModel === 'async')) config.prefetchWarmupFlush = false;
  // Schedules without virtual stages imply vpp = 1; the link does not carry it.
  const sched = SCHEDULES[config.schedule as SimConfig['schedule']];
  if (sched && !sched.supportsVpp) config.vpp = 1;
  const sel = Number(q.get('sel'));
  return { config: config as unknown as SimConfig, selectedMb: q.has('sel') && Number.isFinite(sel) ? sel : null };
}

export function writeUrl(config: SimConfig, defaults: SimConfig, selectedMb: number | null): void {
  const parts: string[] = [];
  const impliedVpp = !SCHEDULES[config.schedule].supportsVpp;
  for (const [key, def] of Object.entries(defaults)) {
    if (key === 'vpp' && impliedVpp) continue;
    if (key === 'groupSize' && (impliedVpp || config.groupSize === config.pp)) continue; // implied
    if (key === 'warmupFormula' && config.schedule !== 'custom') continue; // only Custom uses it
    if (key === 'commModel' && (config.schedule === '1f1b' || config.schedule === 'gpipe')) continue; // implied: blocking-only
    if ((key === 'sendAfter' || key === 'waitGrad') && (config as unknown as Record<string, unknown>)[key] === (megatronPlacement(config.schedule, config.commModel) as unknown as Record<string, unknown>)[key]) continue;
    const v = (config as unknown as Record<string, unknown>)[key];
    if (v !== undefined && v !== def) parts.push(`${key}=${enc(String(v))}`);
  }
  if (config.lengthMode === 'custom' && config.tokens) parts.push(`tokens=${config.tokens.join(',')}`);
  if (selectedMb !== null) parts.push(`sel=${selectedMb}`);
  history.replaceState(null, '', parts.length ? `?${parts.join('&')}` : location.pathname);
}
