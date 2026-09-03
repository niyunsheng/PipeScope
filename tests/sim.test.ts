import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { simulate } from '../src/sim/index.ts';
import type { Op, ScheduleName, SimConfig, Trace } from '../src/sim/index.ts';
import { numWarmup } from '../src/sim/schedules/interleaved.ts';

const EPS = 1e-9;

function cfg(schedule: ScheduleName, pp: number, vpp: number, m: number, extra: Partial<SimConfig> = {}): SimConfig {
  // activationBytes = 1 makes peak memory equal to the number of resident activations.
  return { schedule, pp, vpp, microBatches: m, forwardTime: 1, backwardTime: 2, p2pLatency: 0, activationBytes: 1, ...extra };
}

/** Structural invariants every valid pipeline schedule must satisfy. */
function checkInvariants(trace: Trace): void {
  const { config: c, ops } = trace;
  const byKey = new Map<string, Op>();
  for (const op of ops) {
    const key = `${op.kind}:${op.mb}:${op.stage}`;
    assert.ok(!byKey.has(key), `duplicate op ${key}`);
    byKey.set(key, op);
    assert.ok(op.end > op.start);
  }
  const numStages = c.pp * c.vpp;
  assert.equal(ops.length, 2 * c.microBatches * numStages, 'one F and one B per (mb, stage)');
  for (let mb = 0; mb < c.microBatches; mb++) {
    for (let s = 0; s < numStages; s++) {
      const f = byKey.get(`F:${mb}:${s}`)!;
      const b = byKey.get(`B:${mb}:${s}`)!;
      assert.ok(f && b);
      assert.ok(b.start >= f.end - EPS, `B after F for mb${mb} stage${s}`);
      if (s > 0) {
        const prev = byKey.get(`F:${mb}:${s - 1}`)!;
        assert.ok(f.start >= prev.end + c.p2pLatency - EPS, `F data dep mb${mb} stage${s}`);
      }
      if (s < numStages - 1) {
        const next = byKey.get(`B:${mb}:${s + 1}`)!;
        assert.ok(b.start >= next.end + c.p2pLatency - EPS, `B data dep mb${mb} stage${s}`);
      }
    }
  }
  for (let r = 0; r < c.pp; r++) {
    const mine = ops.filter((o) => o.rank === r).sort((a, b) => a.start - b.start);
    for (let i = 1; i < mine.length; i++) {
      assert.ok(mine[i].start >= mine[i - 1].end - EPS, `ops overlap on rank ${r}`);
    }
  }
  // Memory returns to baseline at the end.
  for (const samples of trace.memory) {
    assert.equal(samples[samples.length - 1].bytes, c.baselineBytes ?? 0);
  }
}

// Peak-memory formulas assume the input is allocated when the forward starts,
// which holds for the sync comm model (and for async when latency is 0 only if
// the sender is not ahead). Use sync so the classic textbook numbers apply.
test('1F1B total time matches closed form when p2p = 0', () => {
  for (const pp of [1, 2, 4, 8]) {
    for (const m of [1, 2, pp, 2 * pp, 3 * pp + 1]) {
      const c = cfg('1f1b', pp, 1, m, { commModel: 'sync' });
      const t = simulate(c);
      checkInvariants(t);
      const expected = (pp - 1) * 3 + m * 3;
      assert.ok(Math.abs(t.metrics.totalTime - expected) < EPS, `pp=${pp} m=${m}: ${t.metrics.totalTime} != ${expected}`);
      // Peak activation memory on rank r is min(pp - r, m).
      t.metrics.ranks.forEach((r) => assert.equal(r.peakMemory, Math.min(pp - r.rank, m)));
    }
  }
});

test('GPipe total time matches closed form and keeps all activations', () => {
  for (const pp of [1, 2, 4]) {
    for (const m of [1, 4, 9]) {
      const t = simulate(cfg('gpipe', pp, 1, m, { commModel: 'sync' }));
      checkInvariants(t);
      assert.ok(Math.abs(t.metrics.totalTime - (m + pp - 1) * 3) < EPS);
      t.metrics.ranks.forEach((r) => assert.equal(r.peakMemory, m));
    }
  }
});

test('Interleaved 1F1B reaches the (pp-1)(tf+tb) bubble bound and Megatron warmup memory', () => {
  for (const pp of [2, 4, 8]) {
    for (const vpp of [1, 2, 3, 4]) {
      for (const m of [pp, 2 * pp, 4 * pp]) {
        for (const tb of [1, 2, 3]) {
          const c = cfg('interleaved-1f1b', pp, vpp, m, { backwardTime: tb, commModel: 'sync' });
          const t = simulate(c);
          checkInvariants(t);
          const expected = (pp - 1) * (1 + tb) + m * vpp * (1 + tb);
          assert.ok(
            Math.abs(t.metrics.totalTime - expected) < EPS,
            `pp=${pp} vpp=${vpp} m=${m} tb=${tb}: ${t.metrics.totalTime} != ${expected}`,
          );
          t.metrics.ranks.forEach((r) => {
            const w = numWarmup(m, pp, r.rank, vpp, pp);
            assert.equal(r.peakMemory, Math.min(w + 1, m * vpp), `peak mem rank ${r.rank}`);
          });
        }
      }
    }
  }
});

test('Interleaved 1F1B rejects micro-batch counts not divisible by pp', () => {
  assert.throws(() => simulate(cfg('interleaved-1f1b', 4, 2, 6)));
});

test('total time and per-rank op order match Victarry/PP-Schedule-Visualization at zero latency', () => {
  // Victarry computes a static longest path without modelling blocking recvs,
  // so individual start times may be earlier than ours (see design.md), but the
  // per-rank op order and the makespan must agree.
  const fixtures = JSON.parse(readFileSync(new URL('./fixtures-victarry.json', import.meta.url), 'utf8')) as Record<
    string,
    { pp: number; vpp: number; m: number; total: number; ops: { rank: number; stage: number; mb: number; kind: 'F' | 'B'; start: number; end: number }[] }
  >;
  for (const [name, fx] of Object.entries(fixtures)) {
    const schedule: ScheduleName = fx.vpp === 1 ? '1f1b' : 'interleaved-1f1b';
    const t = simulate(cfg(schedule, fx.pp, fx.vpp, fx.m));
    assert.ok(Math.abs(t.metrics.totalTime - fx.total) < EPS, `${name} total ${t.metrics.totalTime} vs ${fx.total}`);
    for (let r = 0; r < fx.pp; r++) {
      const mine = t.ops.filter((o) => o.rank === r).sort((a, b) => a.start - b.start).map((o) => `${o.kind}${o.mb}@${o.stage}`);
      const ref = fx.ops.filter((o) => o.rank === r).sort((a, b) => a.start - b.start).map((o) => `${o.kind}${o.mb}@${o.stage}`);
      assert.deepEqual(mine, ref, `${name} rank ${r} op order`);
    }
    // Every op ends no earlier than in the non-blocking reference model.
    const mine = new Map(t.ops.map((o) => [`${o.kind}:${o.mb}:${o.stage}`, o]));
    for (const ref of fx.ops) {
      const o = mine.get(`${ref.kind}:${ref.mb}:${ref.stage}`)!;
      assert.ok(o.end >= ref.end - EPS, `${name} ${ref.kind}${ref.mb}@${ref.stage} earlier than reference`);
    }
  }
});

test('synchronous interleaved warmup blocks on recv_backward before the first steady forward', () => {
  // Megatron's synchronous path issues recv_backward in the last warmup comm and
  // waits for it, so rank 0 cannot start its first steady-state forward until the
  // first gradient arrives from rank pp-1, even though the forward input is ready.
  const t = simulate(cfg('interleaved-1f1b', 4, 2, 8));
  const rank0 = t.ops.filter((o) => o.rank === 0).sort((a, b) => a.start - b.start);
  const firstSteadyF = rank0[numWarmup(8, 4, 0, 2, 4)];
  const firstB = rank0.find((o) => o.kind === 'B')!;
  assert.equal(firstSteadyF.kind, 'F');
  assert.equal(firstSteadyF.mb, 6);
  const grad = t.ops.find((o) => o.kind === 'B' && o.mb === firstB.mb && o.stage === firstB.stage + 1)!;
  assert.ok(Math.abs(firstSteadyF.start - grad.end) < EPS, 'first steady F starts exactly when the gradient arrives');
  assert.ok(firstSteadyF.start > 10, 'later than the non-blocking reference (t=10)');
});

test('p2p latency never speeds things up and all idle time is attributed', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    let prev = 0;
    for (const commModel of ['async', 'sync'] as const) {
      prev = 0;
      for (const lat of [0, 0.1, 0.5, 1, 2]) {
        const t = simulate(cfg(schedule, 4, vpp, 8, { p2pLatency: lat, commModel }));
        checkInvariants(t);
        assert.ok(t.metrics.totalTime >= prev - EPS, `${schedule} ${commModel} lat=${lat} not monotone`);
        prev = t.metrics.totalTime;
        for (const r of t.metrics.ranks) {
          const accounted = r.busy + r.waitRecv + r.waitSend + r.tail;
          assert.ok(Math.abs(accounted - t.metrics.totalTime) < EPS, `${schedule} ${commModel} lat=${lat} rank ${r.rank}: ${accounted} != ${t.metrics.totalTime}`);
        }
      }
    }
  }
});

test('async comm: a recv whose data already landed completes immediately', () => {
  // 1F1B pp4, latency 0.5: rank 0 sends F3 long before rank 1 finishes B0, so
  // under the buffered model rank 1 starts F3 right after B0.
  const t = simulate(cfg('1f1b', 4, 1, 8, { p2pLatency: 0.5, commModel: 'async' }));
  const b0 = t.ops.find((o) => o.rank === 1 && o.kind === 'B' && o.mb === 0)!;
  const f3 = t.ops.find((o) => o.rank === 1 && o.kind === 'F' && o.mb === 3)!;
  assert.ok(Math.abs(f3.start - b0.end) < EPS, `F3 starts at ${f3.start}, B0 ends at ${b0.end}`);
  // Senders never wait under the async model.
  assert.ok(t.idles.every((i) => i.reason === 'wait-recv'));
  // The sync model is never faster than the async one.
  const sync = simulate(cfg('1f1b', 4, 1, 8, { p2pLatency: 0.5, commModel: 'sync' }));
  assert.ok(sync.metrics.totalTime >= t.metrics.totalTime - EPS);
});

test('async comm lands the input on the receiver before its forward starts; sync does not', () => {
  const base = { p2pLatency: 0.5, activationBytes: undefined, seqLen: 1024, hiddenSize: 1024, microBatchSize: 1, dtypeBytes: 2, activationMultiplier: 17, layersPerChunk: 1 };
  const input = 1024 * 1024 * 2;
  for (const commModel of ['async', 'sync'] as const) {
    const t = simulate(cfg('1f1b', 4, 1, 8, { ...base, commModel }));
    // rank 1's F3 input: sent by rank 0 right after F3 (t≈5.5), consumed after B0.
    const f3 = t.ops.find((o) => o.rank === 1 && o.kind === 'F' && o.mb === 3)!;
    const before = t.memory[1].filter((smp) => smp.t < f3.start - EPS);
    const inputEv = t.memory[1].find((smp) => smp.event === 'input' && smp.resident.includes('3:0'))!;
    if (commModel === 'async') {
      assert.ok(inputEv.t < f3.start - EPS, `async: input for mb3 landed at ${inputEv.t}, F3 starts ${f3.start}`);
      assert.ok(before.some((smp) => smp.resident.includes('3:0')));
    } else {
      assert.ok(Math.abs(inputEv.t - f3.start) < EPS, `sync: input lands exactly at F start`);
    }
    // Per (mb, chunk) total is input + 17 * input. On the last rank the sync model
    // holds one activation at a time; async additionally buffers the next input.
    assert.equal(t.metrics.ranks[3].peakMemory, commModel === 'sync' ? 18 * input : 19 * input);
  }
});

test('synchronous 1F1B steady state costs tf + tb + 2 * latency per micro-batch', () => {
  const lat = 0.5;
  const t = simulate(cfg('1f1b', 4, 1, 16, { p2pLatency: lat, commModel: 'sync' }));
  const last = t.ops.filter((o) => o.rank === 3 && o.kind === 'F').sort((a, b) => a.mb - b.mb);
  // Skip the first iteration, which only pays latency once.
  for (let i = 2; i < last.length; i++) {
    assert.ok(Math.abs(last[i].start - last[i - 1].start - (1 + 2 + 2 * lat)) < EPS, `period at mb ${i}`);
  }
});

test('large configurations simulate quickly', () => {
  const t0 = performance.now();
  const t = simulate(cfg('interleaved-1f1b', 16, 4, 256));
  const ms = performance.now() - t0;
  assert.equal(t.ops.length, 2 * 256 * 64);
  assert.ok(ms < 2000, `took ${ms}ms`);
});
