import { test } from 'node:test';
import assert from 'node:assert/strict';
import { simulate, computeScale, quadraticShare } from '../src/sim/index.ts';
import type { SimConfig } from '../src/sim/index.ts';
import { generateLengths, orderLengths } from '../src/ui/lengths.ts';

const EPS = 1e-9;
const base: SimConfig = { schedule: '1f1b', pp: 4, vpp: 1, microBatches: 8, forwardTime: 1, backwardTime: 2, p2pLatency: 0, seqLen: 4096, hiddenSize: 4096 };

test('log-normal lengths are reproducible, keep the mean roughly, and reorder as asked', () => {
  const a = generateLengths({ n: 2000, mean: 4096, mode: 'lognormal', cv: 0.5, seed: 7, order: 'asis' });
  const b = generateLengths({ n: 2000, mean: 4096, mode: 'lognormal', cv: 0.5, seed: 7, order: 'asis' });
  assert.deepEqual(a, b);
  const mean = a.reduce((x, y) => x + y, 0) / a.length;
  assert.ok(Math.abs(mean / 4096 - 1) < 0.001, `mean ${mean}`);
  assert.ok(a.every((x) => x >= 410 && x <= 16384));
  assert.deepEqual(orderLengths([3, 1, 2], 'asc'), [1, 2, 3]);
  assert.deepEqual(orderLengths([3, 1, 2], 'desc'), [3, 2, 1]);
  assert.deepEqual(orderLengths([5, 1, 4, 2, 3], 'alternate'), [5, 1, 4, 2, 3]);
  assert.deepEqual(generateLengths({ n: 3, mean: 4096, mode: 'uniform', cv: 0.5, seed: 1, order: 'desc' }), [4096, 4096, 4096]);
});

test('compute time scales with tokens: linear part plus quadratic attention share', () => {
  const alpha = quadraticShare(base); // 4096 / (6 * 4096 + 4096) = 1/7
  assert.ok(Math.abs(alpha - 1 / 7) < EPS);
  const tokens = [4096, 8192, 2048, 4096, 4096, 4096, 4096, 4096];
  const t = simulate({ ...base, tokens });
  const f = (mb: number) => t.ops.find((o) => o.rank === 0 && o.kind === 'F' && o.mb === mb)!;
  assert.ok(Math.abs(f(0).end - f(0).start - 1) < EPS);
  assert.ok(Math.abs(f(1).end - f(1).start - computeScale(2, alpha)) < EPS);
  assert.ok(Math.abs(f(2).end - f(2).start - computeScale(0.5, alpha)) < EPS);
  // Memory scales linearly: the mb1 activation is exactly twice mb0's.
  const mem = t.memory[3];
  const step = (mb: number) => {
    const ev = mem.find((m) => m.event === 'forward' && m.resident.includes(`${mb}:0`))!;
    const prev = mem[mem.indexOf(ev) - 1];
    return ev.bytes - prev.bytes;
  };
  assert.ok(Math.abs(step(1) / step(0) - 2) < EPS);
});

test('heterogeneous lengths keep schedule invariants and the ideal time equals total compute', () => {
  const tokens = generateLengths({ n: 16, mean: 4096, mode: 'lognormal', cv: 0.8, seed: 3, order: 'alternate' });
  for (const schedule of ['1f1b', 'interleaved-1f1b', 'gpipe'] as const) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    const t = simulate({ ...base, schedule, vpp, microBatches: 16, tokens, p2pLatency: 0.2 });
    assert.equal(t.ops.length, 2 * 16 * 4 * vpp);
    for (const r of t.metrics.ranks) {
      const accounted = r.busy + r.waitRecv + r.waitSend + r.tail;
      assert.ok(Math.abs(accounted - t.metrics.totalTime) < EPS);
      assert.ok(Math.abs(r.busy - t.metrics.idealTime) < EPS, 'all ranks do the same total work');
    }
  }
});

test('linear / attention coefficient ratio k sets the quadratic share s / (k h + s)', () => {
  const tokens = [8192, 4096, 4096, 4096, 4096, 4096, 4096, 4096];
  const cfg = { ...base, tokens, linearAttnRatio: 2 }; // alpha = 4096 / (2 * 4096 + 4096) = 1/3
  assert.ok(Math.abs(quadraticShare(cfg) - 1 / 3) < EPS);
  const t = simulate(cfg);
  const f0 = t.ops.find((o) => o.rank === 0 && o.kind === 'F' && o.mb === 0)!;
  assert.ok(Math.abs(f0.end - f0.start - computeScale(2, 1 / 3)) < EPS); // 2/3*2 + 1/3*4 = 8/3
});

test('tokens length must match the micro-batch count', () => {
  assert.throws(() => simulate({ ...base, tokens: [4096, 4096] }));
});
