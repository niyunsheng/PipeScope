import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { DEFAULT_CONFIG, buildProgram, constantCost, megatronPlacement, parseFormula, runProgram, simulate, validateConfig } from '../src/sim/index.ts';
import type { Op, Program, ScheduleName, SimConfig, Trace } from '../src/sim/index.ts';
import { numWarmup } from '../src/sim/schedules/interleaved.ts';

const EPS = 1e-9;

/** Forward and backward ops only: loss ops always exist but are not part of the F/B accounting. */
const fbOps = (t: Trace): Op[] => t.ops.filter((o) => o.kind !== 'L');

function cfg(schedule: ScheduleName, pp: number, vpp: number, m: number, extra: Partial<SimConfig> = {}): SimConfig {
  // activationBytes = 1 makes peak memory equal to the number of resident activations.
  const commModel = extra.commModel ?? DEFAULT_CONFIG.commModel;
  const base = { ...DEFAULT_CONFIG, ...megatronPlacement(schedule, commModel), schedule, pp, vpp, groupSize: pp, microBatches: m, forwardTime: 1, backwardTime: 2, p2pLatency: 0, activationBytes: 1, ...extra };
  // Megatron's 1F1B always blocks; its idealised non-blocking form is the Custom schedule with one chunk.
  if (schedule === '1f1b' && base.commModel !== 'sync') {
    return { ...base, schedule: 'custom', warmupFormula: 'pp - r - 1', groupSize: m, ...megatronPlacement('1f1b', 'sync'), ...extra };
  }
  return base;
}

/** Structural invariants every valid pipeline schedule must satisfy. */
function checkInvariants(trace: Trace): void {
  const { config: c } = trace;
  const ops = fbOps(trace);
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
  // pp = 2 is excluded on the sync path by Megatron's own rule (see validateConfig).
  for (const pp of [3, 4, 8]) {
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
      const mine = fbOps(t).filter((o) => o.rank === r).sort((a, b) => a.start - b.start).map((o) => `${o.kind}${o.mb}@${o.stage}`);
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
  // The synchronous path is what the sync comm model selects for interleaved;
  // under async Megatron uses the overlap path, where this wait moves to before the B.
  const t = simulate(cfg('interleaved-1f1b', 4, 2, 8, { commModel: 'sync' }));
  const rank0 = t.ops.filter((o) => o.rank === 0).sort((a, b) => a.start - b.start);
  const firstSteadyF = rank0[numWarmup(8, 4, 0, 2, 4)];
  const firstB = rank0.find((o) => o.kind === 'B')!;
  assert.equal(firstSteadyF.kind, 'F');
  assert.equal(firstSteadyF.mb, 6);
  const grad = t.ops.find((o) => o.kind === 'B' && o.mb === firstB.mb && o.stage === firstB.stage + 1)!;
  assert.ok(Math.abs(firstSteadyF.start - grad.end) < EPS, 'first steady F starts exactly when the gradient arrives');
  assert.ok(firstSteadyF.start > 10, 'later than the non-blocking reference (t=10)');
  // Overlap path (async): the first steady forward is not held up; the wait sits before the backward.
  const ov = simulate(cfg('interleaved-1f1b', 4, 2, 8, { commModel: 'async' }));
  const r0 = ov.ops.filter((o) => o.rank === 0).sort((a, b) => a.start - b.start);
  assert.ok(Math.abs(r0[numWarmup(8, 4, 0, 2, 4)].start - 10) < EPS, 'overlap: first steady F starts right after warmup');
});

test('p2p latency never speeds things up and all idle time is attributed', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    let prev = 0;
    for (const commModel of ['async', 'sync'] as const) {
      prev = 0;
      for (const lat of [0, 0.1, 0.5, 1, 2]) {
        const t = simulate(cfg(schedule, 4, vpp, 8, { p2pLatency: lat, commModel }));
        assert.equal(t.failure, null);
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

test('predecessors: every op names predecessors that finished first; the chain explains the warmup bubble', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    for (const commModel of ['async', 'sync'] as const) {
      const t = simulate(cfg(schedule, 4, vpp, 8, { p2pLatency: 0.5, commModel }));
      for (const op of t.ops) {
        assert.equal(t.ops[op.id], op, 'ids index into ops');
        if (!op.predecessors.length) {
          assert.equal(op.start, 0, `${schedule}: only ops at t=0 may lack a predecessor`);
          continue;
        }
        for (const b of op.predecessors) {
          const pred = t.ops[b.op];
          assert.ok(pred.end <= op.start + EPS, `${schedule} ${commModel}: predecessor ends after op starts`);
          if (b.reason !== 'program') {
            // A wait names the transfer that released the rank, and that transfer touches both ranks involved.
            const tr = t.transfers.find((x) => x.tag === b.transferTag);
            assert.ok(tr, `${schedule} ${commModel}: wait predecessor without a transfer`);
            assert.ok([tr!.from, tr!.to].includes(op.rank) && [tr!.from, tr!.to].includes(pred.rank));
            assert.ok(tr!.landed <= op.start + EPS || b.reason === 'wait-send');
          }
          if (b.reason === 'program') {
            assert.equal(pred.rank, op.rank);
            assert.ok(Math.abs(pred.end - op.start) < EPS, 'program order binds only when the previous op ended exactly at start');
          }
        }
        // Cross-rank waits carry pipeline information, so they come first.
        const firstProgram = op.predecessors.findIndex((b) => b.reason === 'program');
        assert.ok(firstProgram === -1 || firstProgram === op.predecessors.length - 1);
      }
    }
  }
  // Default config: rank 0's first steady forward waits for the gradient of
  // mb0 to come back through ranks 1 and 2, each doing one F and one B.
  // Synchronous-path placement under the async comm model: the classic "rank 0 waits
  // from 10 to 16 for mb0's gradient" picture (Custom pins the placement explicitly).
  const t = simulate(cfg('custom', 4, 2, 8, { commModel: 'async', sendAfter: 'B', waitGrad: 'beforeF' }));
  const f6 = t.ops.find((o) => o.rank === 0 && o.kind === 'F' && o.mb === 6 && o.chunk === 0)!;
  const chain: Op[] = [];
  for (let cur: Op | null = f6; cur; cur = cur.predecessors.length ? t.ops[cur.predecessors[0].op] : null) chain.push(cur);
  const head = chain.slice(0, 6).map((o) => `r${o.rank}:${o.kind}${o.mb}c${o.chunk}:${o.predecessors[0].reason}`);
  assert.deepEqual(head, [
    'r0:F6c0:wait-recv',
    'r1:B0c1:program',
    'r1:F4c0:wait-recv',
    'r2:B0c1:program',
    'r2:F2c1:wait-recv',
    'r3:B0c1:program',
  ]);
  assert.equal(chain[chain.length - 1].start, 0, 'chain reaches t = 0');
  // Tie: rank 3's F mb0 c1 (stage 7) starts at 7 both because its own F mb3 c0
  // ended at 7 and because the stage-6 output from rank 2 landed at 7.
  const f0c1 = t.ops.find((o) => o.rank === 3 && o.kind === 'F' && o.mb === 0 && o.chunk === 1)!;
  const preds = f0c1.predecessors.map((b) => `${b.reason}:r${t.ops[b.op].rank}:${t.ops[b.op].kind}${t.ops[b.op].mb}c${t.ops[b.op].chunk}`);
  assert.deepEqual(preds, ['wait-recv:r2:F0c1', 'program:r3:F3c0']);
});

test('transfers: one record per received tensor, consistent with ops, posting order and landing times', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    for (const commModel of ['async', 'sync'] as const) {
      const lat = 0.2;
      const t = simulate(cfg(schedule, 4, vpp, 8, { p2pLatency: lat, commModel }));
      const numStages = 4 * vpp;
      // Every stage boundary is crossed once by an activation and once by a gradient.
      assert.equal(t.transfers.length, 2 * 8 * (numStages - 1));
      assert.equal(new Set(t.transfers.map((x) => x.tag)).size, t.transfers.length, 'tags unique');
      for (const tr of t.transfers) {
        assert.ok(Math.abs(tr.landed - tr.start - lat) < EPS);
        const earliest = commModel === 'sync' ? Math.max(tr.sendPosted, tr.recvPosted) : tr.sendPosted;
        assert.ok(Math.abs(tr.start - earliest) < EPS, `${schedule} ${commModel}: data starts when ${commModel === 'sync' ? 'both posted' : 'sent'}`);
        assert.notEqual(tr.producer, null);
        const prod = t.ops[tr.producer!];
        assert.equal(prod.rank, tr.from);
        assert.equal(prod.mb, tr.mb);
        assert.ok(prod.end <= tr.sendPosted + EPS, 'producer finished before the send was posted');
        // The consumer is the op on the receiving rank whose input is this tensor; it cannot start before landing.
        const consumer = t.ops.find((o) => o.rank === tr.to && o.mb === tr.mb && o.kind === tr.kind && (tr.kind === 'F' ? o.stage === prod.stage + 1 : o.stage === prod.stage - 1))!;
        assert.ok(consumer, 'consumer exists');
        assert.ok(consumer.start >= tr.landed - EPS, `${schedule} ${commModel}: consumer starts before its input landed`);
      }
    }
  }
});

test('loss ops sit between the last-stage forward and backward and do not touch memory', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    const base = simulate(cfg(schedule, 4, vpp, 8));
    const t = simulate(cfg(schedule, 4, vpp, 8, { lossTime: 0.5 }));
    const last = 4 * vpp - 1;
    const losses = t.ops.filter((o) => o.kind === 'L');
    assert.equal(losses.length, 8, 'one loss per micro-batch');
    for (const l of losses) {
      assert.equal(l.stage, last);
      assert.ok(Math.abs(l.end - l.start - 0.5) < EPS);
      const f = t.ops.find((o) => o.kind === 'F' && o.mb === l.mb && o.stage === last)!;
      const b = t.ops.find((o) => o.kind === 'B' && o.mb === l.mb && o.stage === last)!;
      assert.ok(Math.abs(f.end - l.start) < EPS, 'loss starts when the last forward ends');
      assert.ok(l.end <= b.start + EPS, 'backward waits for the loss');
    }
    // Eight losses on the last rank cost at least 8 * 0.5 of wall-clock.
    assert.ok(t.metrics.totalTime >= base.metrics.totalTime + 4 - EPS);
    // Memory events are only F allocations and B releases.
    assert.equal(t.memory[3].filter((s) => s.event === 'release').length, base.memory[3].filter((s) => s.event === 'release').length);
    assert.equal(t.metrics.ranks[3].peakMemory, base.metrics.ranks[3].peakMemory);
  }
});

test('custom schedule: defaults reproduce Megatron; the warmup formula is live', () => {
  const key = (t: Trace) => JSON.stringify(t.ops.map((o) => [o.rank, o.kind, o.mb, o.chunk, o.start, o.end]).sort());
  const mega = simulate(cfg('interleaved-1f1b', 4, 2, 8, { p2pLatency: 0.2 }));
  const same = simulate(cfg('custom', 4, 2, 8, { p2pLatency: 0.2 }));
  assert.equal(same.failure, null);
  assert.equal(key(same), key(mega), 'default knobs == Megatron interleaved');

  // Halving the first term keeps the program legal but the pipeline starves (see learning doc, step 3).
  const unit = { forwardTime: 1, backwardTime: 1 };
  const megaUnit = simulate(cfg('interleaved-1f1b', 4, 2, 8, unit));
  const sync = { sendAfter: 'B', waitGrad: 'beforeF' } as const;
  const half = simulate(cfg('custom', 4, 2, 8, { ...unit, ...sync, warmupFormula: '(pp - r - 1) + (vpp - 1) * G' }));
  assert.equal(half.failure, null);
  assert.equal(megaUnit.metrics.totalTime, 38);
  assert.equal(half.metrics.totalTime, 60);
  // Same experiment on the interleaved schedule itself under sync (its synchronous path).
  assert.equal(simulate(cfg('interleaved-1f1b', 4, 2, 8, { ...unit, commModel: 'sync' })).metrics.totalTime, 38);

  // One fewer than Megatron breaks the last rank: B before its own F, reported as a program failure.
  const less = simulate(cfg('custom', 4, 2, 8, { ...sync, warmupFormula: '2 * (pp - r - 1) + (vpp - 1) * G - 1' }));
  assert.equal(less.failure?.kind, 'program');
  assert.equal(less.failure?.op?.rank, 3);

  // Sending the forward output right after F recovers part of the loss (the
  // recv for the next forward is still posted after B), not all of it.
  const fast = simulate(cfg('custom', 4, 2, 8, { ...unit, warmupFormula: '(pp - r - 1) + (vpp - 1) * G', sendAfter: 'F', waitGrad: 'beforeF' }));
  assert.equal(fast.failure, null);
  assert.ok(fast.metrics.totalTime < half.metrics.totalTime && fast.metrics.totalTime > megaUnit.metrics.totalTime, `after-F send: ${fast.metrics.totalTime}`);

  // Overlap placement: with vpp = 1 and a = 1 it is exactly 1F1B (same ops, same times);
  // with vpp = 2 the 1x warmup now reaches Megatron's bubble, so the factor 2 is the
  // price of the batched placement, not of interleaving itself.
  const overlap = { sendAfter: 'F', waitGrad: 'beforeB' } as const;
  const ov1 = simulate(cfg('custom', 4, 1, 8, { ...unit, warmupFormula: '(pp - r - 1) + (vpp - 1) * G', ...overlap }));
  const f1b = simulate(cfg('1f1b', 4, 1, 8, unit));
  assert.equal(ov1.failure, null);
  assert.equal(key(ov1), key(f1b), 'overlap + a=1 + vpp=1 == 1F1B');
  const ov2 = simulate(cfg('custom', 4, 2, 8, { ...unit, warmupFormula: '(pp - r - 1) + (vpp - 1) * G', ...overlap }));
  assert.equal(ov2.failure, null);
  assert.equal(ov2.metrics.totalTime, megaUnit.metrics.totalTime, `overlap: ${ov2.metrics.totalTime}`);
  assert.ok(ov2.metrics.ranks[0].peakMemory < megaUnit.metrics.ranks[0].peakMemory, 'and less memory');

  // Non-default group size accepted by Megatron's rules; m = 10, G = 5 with pp = 4 (10 mod 5 = 0).
  const g5 = simulate(cfg('custom', 4, 2, 10, { groupSize: 5 }));
  assert.equal(g5.failure, null);
  checkInvariants(g5);
});

test('custom schedule: validation rejects bad formulas and group sizes with a clear message', () => {
  const bad = (extra: Partial<SimConfig>) => validateConfig(cfg('custom', 4, 2, 8, extra));
  assert.match(bad({ warmupFormula: '2 * (pp - r' }).join(' '), /warmup formula: expected '\)'/);
  assert.match(bad({ warmupFormula: 'pp * q' }).join(' '), /unknown variable q/);
  assert.match(bad({ groupSize: 3 }).join(' '), /group size must be/);
  assert.match(bad({ groupSize: 5 }).join(' '), /m mod G = 3 must be 0 or ≥ pp/);
  // The same rule now governs the Megatron schedule itself.
  assert.match(validateConfig(cfg('interleaved-1f1b', 4, 2, 9, {})).join(' '), /m mod G = 1 must be 0/);
  assert.deepEqual(validateConfig(cfg('interleaved-1f1b', 4, 2, 10, { groupSize: 5 })), []);
  assert.equal(simulate(cfg('interleaved-1f1b', 4, 2, 10, { groupSize: 5 })).failure, null);
  // Deadlock-prone combinations are not pre-judged (the engine reports them); a non-default G under sync passes validation.
  assert.deepEqual(validateConfig(cfg('interleaved-1f1b', 4, 2, 10, { groupSize: 5, commModel: 'sync' })), []);
  assert.equal(simulate(cfg('interleaved-1f1b', 4, 2, 10, { groupSize: 5, commModel: 'sync' })).failure?.kind, 'deadlock');
  assert.equal(simulate(cfg('interleaved-1f1b', 4, 2, 9, { groupSize: 5, commModel: 'sync' })).failure, null);
  // Megatron's own rule: interleaved without overlap needs pp > 2.
  assert.match(validateConfig(cfg('interleaved-1f1b', 2, 2, 4, { commModel: 'sync' })).join(' '), /pp > 2/);
  assert.deepEqual(validateConfig(cfg('interleaved-1f1b', 2, 2, 4, { commModel: 'async' })), []);
  // A blocking send right after F would rendezvous with a recv that is only posted after B: deadlock by construction.
  assert.match(bad({ sendAfter: 'F', commModel: 'sync' }).join(' '), /only "send after B"/);
  assert.match(bad({ waitGrad: 'beforeB', commModel: 'sync' }).join(' '), /only "send after B"/);
  // Built-in schedules only accept Megatron's placement for the chosen comm model; Custom is free.
  assert.match(validateConfig(cfg('1f1b', 4, 1, 8, { sendAfter: 'B', waitGrad: 'beforeF', commModel: 'sync' })).join(' '), /use the Custom schedule/);
  assert.match(validateConfig({ ...cfg('1f1b', 4, 1, 8, { commModel: 'sync' }), commModel: 'async' }).join(' '), /1F1B runs under the sync comm model only/);
  assert.match(validateConfig(cfg('interleaved-1f1b', 4, 2, 8, { sendAfter: 'B', waitGrad: 'beforeF', commModel: 'async' })).join(' '), /use the Custom schedule/);
  assert.deepEqual(validateConfig(cfg('interleaved-1f1b', 4, 2, 8, { commModel: 'sync' })), []);
  assert.deepEqual(validateConfig(cfg('custom', 4, 1, 8, { sendAfter: 'B', waitGrad: 'beforeF', commModel: 'sync' })), []);
  assert.deepEqual(bad({}), []);
});

test('formula parser: precedence, unary minus, functions, comparisons, error positions', () => {
  const ev = (src: string, vars = {}) => parseFormula(src).eval(vars);
  assert.equal(ev('1 + 2 * 3'), 7);
  assert.equal(ev('(1 + 2) * 3'), 9);
  assert.equal(ev('-2 * -3'), 6);
  assert.equal(ev('7 % 4'), 3);
  assert.equal(ev('min(3, 1, 2) + max(1, 5) + floor(2.7) + ceil(2.1) + abs(-1)'), 1 + 5 + 2 + 3 + 1);
  assert.equal(ev('(r == pp - 1) * 10', { r: 3, pp: 4 }), 10);
  assert.equal(ev('(r == pp - 1) * 10', { r: 2, pp: 4 }), 0);
  assert.deepEqual(parseFormula('2 * (pp - r - 1) + (vpp - 1) * G').vars.sort(), ['G', 'pp', 'r', 'vpp']);
  assert.throws(() => parseFormula('1 +'), /unexpected end of formula \(at position 4\)/);
  assert.throws(() => parseFormula('1 $ 2'), /unexpected '\$' \(at position 3\)/);
  assert.throws(() => parseFormula('$'), /unexpected character '\$' \(at position 1\)/);
  assert.throws(() => parseFormula('foo(1)'), /unknown function foo/);
});

test('sync-model deadlock with non-default group size follows the drift rule floor(m/G)*(G-pp) >= 2 (with a steady state)', () => {
  // Bypass validation: run the raw program and compare the engine's verdict with the rule.
  let cases = 0;
  for (const pp of [2, 3, 4]) {
    for (let m = pp; m <= 20; m++) {
      for (let G = pp + 1; G <= m; G++) {
        if (m % G !== 0 && m % G < pp) continue; // Megatron's own constraint
        const c = { ...cfg('interleaved-1f1b', pp, 2, m, { commModel: 'sync' }), groupSize: G };
        const r = runProgram(buildProgram(c), pp, constantCost(c), 'sync');
        const steadyRank0 = m * 2 - numWarmup(m, pp, 0, 2, G);
        const predicted = Math.floor(m / G) * (G - pp) >= 2 && steadyRank0 >= 2;
        assert.equal(r.failure !== null, predicted, `pp=${pp} m=${m} G=${G}`);
        if (r.failure) assert.equal(r.failure.kind, 'deadlock');
        cases++;
      }
    }
  }
  assert.ok(cases > 100);
});

test('a program that schedules a backward before its own forward is rejected', () => {
  // Hand-built two-rank program: rank 1 (last stage) runs B before F.
  const bad = [
    [{ type: 'compute', kind: 'F', mb: 0, chunk: 0 }, { type: 'comm', sends: [{ kind: 'F', peer: 1, tag: 'F:0:0', mb: 0 }], recvs: [] }],
    [{ type: 'comm', sends: [], recvs: [{ kind: 'F', peer: 0, tag: 'F:0:0', mb: 0 }] }, { type: 'compute', kind: 'B', mb: 0, chunk: 0 }, { type: 'compute', kind: 'F', mb: 0, chunk: 0 }],
  ] as Program;
  const cost = constantCost({ ...DEFAULT_CONFIG, pp: 2, vpp: 1, microBatches: 1 });
  const r = runProgram(bad, 2, cost, 'async');
  assert.equal(r.failure?.kind, 'program');
  assert.match(r.failure!.message, /Program error on rank 1/);
  // What ran before the failure is kept, and the illegal step is reported where it would have started.
  assert.equal(r.ops.length, 1);
  assert.equal(r.failure!.op?.rank, 1);
  assert.equal(r.failure!.op?.kind, 'B');
  assert.ok(Math.abs(r.failure!.op!.start - 1.2) < EPS, 'B would have started when the input landed');
});

test('a deadlocked program reports the stuck ranks and keeps the partial timeline', () => {
  // Rank 0 waits for a tensor rank 1 never sends.
  const bad = [
    [{ type: 'compute', kind: 'F', mb: 0, chunk: 0 }, { type: 'comm', sends: [], recvs: [{ kind: 'B', peer: 1, tag: 'B:0:1', mb: 0 }] }],
    [{ type: 'compute', kind: 'F', mb: 0, chunk: 0 }],
  ] as Program;
  const r = runProgram(bad, 2, constantCost({ ...DEFAULT_CONFIG, pp: 2, vpp: 1, microBatches: 1 }), 'async');
  assert.equal(r.failure?.kind, 'deadlock');
  assert.deepEqual(r.failure!.blocked.map((b) => b.rank), [0]);
  assert.equal(r.failure!.blocked[0].since, 1);
  assert.equal(r.ops.length, 2);
});

test('every idle names an existing transfer that involves the idle rank and ends when the idle ends', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    for (const commModel of ['async', 'sync'] as const) {
      const t = simulate(cfg(schedule, 4, vpp, 8, { p2pLatency: 0.5, commModel }));
      for (const i of t.idles) {
        const tr = t.transfers.find((x) => x.tag === i.transferTag)!;
        assert.ok(tr, 'idle without transfer');
        assert.equal(i.reason === 'wait-recv' ? tr.to : tr.from, i.rank);
        // The wait ends when the data lands (recv) or, for a blocking send, when the transfer completes.
        assert.ok(Math.abs(tr.landed - i.end) < EPS, `${schedule} ${commModel}: idle ends at ${i.end}, transfer lands at ${tr.landed}`);
      }
    }
  }
});

test('idle breakdown: peerWait + transfer covers the whole idle; async == sync at zero latency', () => {
  for (const schedule of ['gpipe', '1f1b', 'interleaved-1f1b'] as ScheduleName[]) {
    const vpp = schedule === 'interleaved-1f1b' ? 2 : 1;
    for (const commModel of ['async', 'sync'] as const) {
      for (const lat of [0, 0.5]) {
        const t = simulate(cfg(schedule, 4, vpp, 8, { p2pLatency: lat, commModel }));
        for (const i of t.idles) {
          assert.ok(i.peerWait >= -EPS && i.transfer >= -EPS, `${schedule} ${commModel} negative component`);
          assert.ok(Math.abs(i.peerWait + i.transfer - (i.end - i.start)) < EPS, `${schedule} ${commModel} lat=${lat}: ${i.peerWait} + ${i.transfer} != ${i.end - i.start}`);
          if (lat === 0) assert.ok(Math.abs(i.transfer) < EPS, 'no wire time at zero latency');
        }
      }
    }
  }
  // Non-interleaved schedules never make a sender wait, so at zero latency the
  // two comm models must agree on everything, including the idle breakdown.
  const key = (t: Trace) => JSON.stringify(t.idles.map((i) => [i.rank, i.start, i.end, i.reason, i.peerWait, i.transfer]).sort());
  for (const schedule of ['gpipe', '1f1b'] as ScheduleName[]) {
    const a = simulate(cfg(schedule, 4, 1, 8, { commModel: 'async' }));
    const s = simulate(cfg(schedule, 4, 1, 8, { commModel: 'sync' }));
    assert.equal(key(a), key(s), `${schedule}: async and sync idles differ at zero latency`);
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

test('input buffers are allocated when the recv is posted (or data lands), before the forward starts', () => {
  const base = { p2pLatency: 0.5, activationBytes: undefined, seqLen: 1024, hiddenSize: 1024, microBatchSize: 1, dtypeBytes: 2, activationMultiplier: 17, layersPerChunk: 1 };
  const input = 1024 * 1024 * 2;
  for (const commModel of ['async', 'sync'] as const) {
    const t = simulate(cfg('1f1b', 4, 1, 8, { ...base, commModel }));
    // rank 1's F3 input: sent by rank 0 right after F3 (t≈5.5), consumed after B0.
    const f3 = t.ops.find((o) => o.rank === 1 && o.kind === 'F' && o.mb === 3)!;
    const before = t.memory[1].filter((smp) => smp.t < f3.start - EPS);
    const inputEv = t.memory[1].find((smp) => smp.event === 'input' && smp.resident.includes('3:0'))!;
    const tr = t.transfers.find((x) => x.to === 1 && x.kind === 'F' && x.mb === 3)!;
    assert.ok(Math.abs(inputEv.t - Math.min(tr.recvPosted, tr.start)) < EPS, `${commModel}: input allocated at ${inputEv.t}`);
    assert.ok(inputEv.t <= f3.start + EPS);
    if (commModel === 'async') {
      // Data was buffered before rank 1 even posted the recv.
      assert.ok(tr.landed < tr.recvPosted - EPS);
      assert.ok(before.some((smp) => smp.resident.includes('3:0')));
    } else {
      // Under sync the buffer exists from the recv posting; data lands at F start.
      assert.ok(Math.abs(tr.landed - f3.start) < EPS);
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
  assert.equal(fbOps(t).length, 2 * 256 * 64);
  assert.ok(ms < 2000, `took ${ms}ms`);
});
