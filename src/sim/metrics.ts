import type { IdleInterval, MemorySample, Metrics, Op, RankMetrics } from './types.ts';

export function computeMetrics(
  ops: Op[],
  idles: IdleInterval[],
  memory: MemorySample[][],
  rankFinish: number[],
  pp: number,
): Metrics {
  const totalTime = Math.max(0, ...rankFinish, ...ops.map((o) => o.end));
  const ranks: RankMetrics[] = [];
  for (let r = 0; r < pp; r++) {
    const busy = ops.filter((o) => o.rank === r).reduce((s, o) => s + (o.end - o.start), 0);
    const waitRecv = idles.filter((i) => i.rank === r && i.reason === 'wait-recv').reduce((s, i) => s + (i.end - i.start), 0);
    const waitSend = idles.filter((i) => i.rank === r && i.reason === 'wait-send').reduce((s, i) => s + (i.end - i.start), 0);
    ranks.push({
      rank: r,
      busy,
      waitRecv,
      waitSend,
      tail: totalTime - rankFinish[r],
      utilization: totalTime > 0 ? busy / totalTime : 0,
      peakMemory: Math.max(0, ...memory[r].map((s) => s.bytes)),
    });
  }
  // With homogeneous per-chunk costs every rank has the same compute total;
  // use the max so heterogeneous v2 costs still give a meaningful lower bound.
  const idealTime = Math.max(0, ...ranks.map((r) => r.busy));
  const totalBusy = ranks.reduce((s, r) => s + r.busy, 0);
  return {
    totalTime,
    idealTime,
    overheadRatio: idealTime > 0 ? (totalTime - idealTime) / idealTime : 0,
    bubbleFraction: totalTime > 0 ? 1 - totalBusy / (pp * totalTime) : 0,
    ranks,
  };
}
