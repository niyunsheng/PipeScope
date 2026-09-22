import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_CONFIG, activationBytes, constantCost, inputBytes, simulate, validateConfig } from '../src/sim/index.ts';

const cfg = { ...DEFAULT_CONFIG, vpp: 10, microBatches: 32, p2pLatency: 0, seqLen: 7103, numLayers: 80 };
test('chunk input is counted once in total activation, including the input-only boundary', () => {
  const input = inputBytes(cfg);
  assert.deepEqual(activationBytes(cfg), { input, intermediate: 33 * input });
  assert.deepEqual(activationBytes({ ...cfg, activationMultiplier: 0.5 }), { input, intermediate: 0 });
  assert.ok(validateConfig({ ...cfg, activationMultiplier: 0.49 }).some(e => e.includes('activation multiplier')));
  assert.ok(validateConfig({ ...cfg, activationMultiplier: Infinity }).some(e => e.includes('activation multiplier')));
  assert.deepEqual(activationBytes({ ...cfg, activationBytes: 10, activationMultiplier: 0 }), { input: 10, intermediate: 0 });
});

test('reference workload peak is 43 chunk activations plus three input-sized buffers', () => {
  const trace = simulate(cfg);
  assert.equal(trace.failure, null);
  assert.equal(trace.metrics.totalTime, 1938);
  assert.equal(trace.metrics.ranks[0].peakMemory, (43 * 34 + 3) * inputBytes(cfg));
  assert.equal(trace.metrics.ranks[0].peakMemory, 85245091840);
  for (const samples of trace.memory) {
    assert.ok(samples.every(s => s.bytes >= 0));
    assert.equal(samples.at(-1)!.bytes, 0);
  }
});

test('one micro-batch allocates input then remaining activation and releases both, including variable lengths', () => {
  const c = { ...DEFAULT_CONFIG, schedule: 'custom' as const, pp: 1, vpp: 1, groupSize: 1, numLayers: 2, microBatches: 1, warmupFormula: '0', tokens: [DEFAULT_CONFIG.seqLen * 2], baselineBytes: 100 };
  const input = inputBytes(c) * 2;
  const trace = simulate(c);
  assert.deepEqual(trace.memory[0].map(s => [s.event, s.bytes]), [['release', 100], ['input', 100 + input], ['forward', 100 + 34 * input], ['release', 100]]);
  const oldCost = constantCost(c);
  const old = simulate(c, { ...oldCost, activationIntermediate: (mb, chunk, rank) => oldCost.activationIntermediate(mb, chunk, rank) + oldCost.activationInput(mb, chunk, rank) });
  assert.deepEqual(trace.ops, old.ops);
  assert.deepEqual(trace.transfers, old.transfers);
});
