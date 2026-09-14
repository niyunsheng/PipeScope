import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CONFIG, simulate, buildProgram, validateConfig } from '../src/sim/index.ts';
import { moeTiming } from '../src/sim/moe.ts';
import { constantCost } from '../src/sim/cost.ts';

const base = { ...DEFAULT_CONFIG, schedule: 'custom' as const, pp: 1, vpp: 1, groupSize: 1, numLayers: 1, microBatches: 4, warmupFormula: '0', p2pLatency: 0, moeOverlap: true, warmupPlusOne: true };
test('fixed eight-node schedule: per-pass events and FIFO streams determine timing', () => {
  const cfg = { ...base, forwardTime: 4, backwardTime: 4, moeRatios: '25/25/25/25' };
  const result = moeTiming({ type: 'compute', kind: 'F', mb: 1, chunk: 0, backward: { mb: 0, chunk: 0 } }, 0, 0, cfg, constantCost(cfg));
  assert.equal(result.end, 4);
  assert.deepEqual(result.segments[0].map(s => [s.part, s.start, s.end]), [['attn', 0, 1], ['dispatch', 1, 2], ['experts', 2, 3], ['combine', 3, 4]]);
  assert.deepEqual(result.segments[1].map(s => [s.part, s.start, s.end]), [['combine', 0, 1], ['experts', 1, 2], ['dispatch', 2, 3], ['attn', 3, 4]]);
});
test('no warmup: fixed pair cannot backward through its unfinished forward', () => {
  const trace = simulate({ ...base, warmupPlusOne: false });
  assert.equal(trace.failure?.kind, 'program');
  assert.match(trace.failure!.message, /combine backward/);
  assert.equal(trace.ops.length, 0);
});
test('extra warmup enables independent pairs, with no compute overlap or double counting', () => {
  const trace = simulate(base);
  assert.equal(trace.failure, null);
  assert.equal(trace.ops.filter(o => o.kind === 'F').length, 4);
  assert.equal(trace.ops.filter(o => o.kind === 'B').length, 4);
  const segments = trace.ops.flatMap(o => o.segments ?? []).filter(s => s.resource === 'compute').sort((a,b) => a.start - b.start);
  for (let i = 1; i < segments.length; i++) assert.ok(segments[i].start >= segments[i-1].end - 1e-9);
  assert.ok(Math.abs(trace.metrics.ranks[0].busy - 4 * 3 * 0.7) < 1e-9);
  assert.ok(trace.memory[0].every(s => s.bytes >= 0));
  assert.equal(trace.memory[0].at(-1)!.bytes, 0);
});
test('warmup +1 alone preserves the number of passes and clamps to total', () => {
  for (const warmupFormula of ['0', 'total']) {
    const trace = simulate({ ...base, moeOverlap: false, warmupFormula });
    assert.equal(trace.failure, null);
    assert.equal(trace.ops.filter(o => o.kind === 'F').length, 4);
    assert.equal(trace.ops.filter(o => o.kind === 'B').length, 4);
  }
});
test('ratio validation and fixed partners are explicit', () => {
  for (const moeRatios of ['1/2/3', '30/15/40/16', 'NaN/0/0/100', '-1/1/50/50']) assert.ok(validateConfig({ ...base, moeRatios }).length);
  const pairs = buildProgram(base)[0].filter(s => s.type === 'compute' && s.backward);
  assert.equal(pairs.length, 3);
});
test('multi-layer forward ascending, backward descending', () => {
  const trace = simulate({ ...base, numLayers: 3 });
  assert.equal(trace.failure, null);
  const f = trace.ops.find(o => o.pair !== undefined && o.kind === 'F')!;
  const b = trace.ops.find(o => o.pair === f.pair && o.kind === 'B')!;
  assert.deepEqual(f.segments!.filter(s => s.part === 'attn').map(s => s.layer), [0,1,2]);
  assert.deepEqual(b.segments!.filter(s => s.part === 'attn').map(s => s.layer), [2,1,0]);
});

test('VPP pipeline preserves tensor causality, partial failure and loss dependencies', () => {
  const cfg = { ...DEFAULT_CONFIG, moeOverlap: true, warmupPlusOne: true, lossTime: 0.2, numLayers: 16 };
  const t = simulate(cfg);
  assert.equal(t.failure, null);
  for (const tr of t.transfers) {
    assert.notEqual(tr.producer, null);
    assert.ok(t.ops[tr.producer!].end <= tr.sendPosted + 1e-9);
    const stage = t.ops[tr.producer!].stage + (tr.kind === 'F' ? 1 : -1);
    const consumer = t.ops.find(o => o.stage === stage && o.kind === tr.kind && o.mb === tr.mb)!;
    assert.ok(consumer.start >= tr.landed - 1e-9);
  }
  for (const loss of t.ops.filter(o => o.kind === 'L')) {
    const b = t.ops.find(o => o.kind === 'B' && o.stage === loss.stage && o.mb === loss.mb)!;
    assert.ok(b.start >= loss.end - 1e-9);
  }
  const failed = simulate({ ...cfg, warmupPlusOne: false });
  assert.ok(failed.failure);
  assert.ok(failed.ops.length > 0);
  assert.ok(failed.ops.length < t.ops.length);
});


test('PP=1 VPP schedule uses the formula; one warmup requires the +1 switch', () => {
  const cfg = { ...base, schedule: 'interleaved-1f1b' as const, warmupPlusOne: false };
  const failed = simulate(cfg);
  assert.equal(failed.failure?.kind, 'program');
  assert.equal(failed.ops.length, 0);
  const trace = simulate({ ...cfg, warmupPlusOne: true });
  assert.equal(trace.failure, null);
  assert.equal(trace.ops.filter(o => o.kind === 'F' && o.pair === undefined).length, 1);
  assert.equal(trace.ops.filter(o => o.kind === 'F' && o.pair !== undefined).length, 3);
  assert.equal(trace.transfers.length, 0);
});
