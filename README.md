# PipeScope

**Live demo: <https://niyunsheng.github.io/PipeScope/>**

Pipeline-parallel (PP) schedule visualizer built on a discrete-event simulator. Ranks run their Megatron-style programs step by step and communication blocks exactly where it would in practice, so timelines reflect real behaviour under p2p latency, sync vs. async communication, and other scenarios instead of idealized diagrams.

Shows GPipe / 1F1B / interleaved 1F1B (VPP) / V-shaped 1F1B timelines per rank and chunk, the full path of any micro-batch, per-rank activation memory, and bubble attribution.

Pure TypeScript, zero npm dependencies (Node.js ≥ 22.18).

```bash
npm test          # simulator tests
npm run build     # build to dist/
npm run serve     # preview at http://localhost:5173/
```

MIT

### MoE A2A overlap

The existing Pipeline panel has a shared `Attn / Dispatch / Experts / Combine`
percentage input (sum 100), `Warmup +1`, and `1F1B overlap`. The switches are
independent and are included in shared URLs. Forward and backward times are the
unoverlapped totals, including A2A, when the MoE model is enabled. Backward uses
the same percentages in reverse dependency order.

The simulator follows `TransformerLayerSchedulePlan.run` from NVIDIA Megatron-LM
commit `630956b357d4b2375e3fc5c7be8b8e429092866d`, with delayed wgrad and early
attention release disabled. For each fixed F/B layer pair it issues:

1. B combine
2. F attention
3. B experts
4. F dispatch
5. B dispatch
6. F experts
7. F combine
8. B attention

Compute and EP communication each have one FIFO stream. Each node also waits for
the preceding node of its own pass. Forward layers ascend and backward layers
descend. No partner selection, dynamic reordering, or serial fallback is used.
Both PP inputs are ready before the pair starts; P2P outputs are posted after
the combined step, using the selected blocking/nonblocking transport. This
first model does not reproduce Megatron's P2P overlap within the combined step.
It assumes balanced EP ranks and no compute/network contention; EP peer skew,
shared experts, MTP, delayed wgrad, and extra dispatch buffers are not modeled.
Existing token-length scaling is applied to all four component costs.

Steady F/B pairs have a dashed outline. Overlapped pairs show F above B, with
the original micro-batch number labels; chunk identity remains in the colors
and tooltip. Each
pass is drawn as one solid block; internal compute/communication events affect
timing without subdividing the block. Hover shows component costs. Utilization counts compute segments only. Activation
memory retains the existing chunk-level allocation model. Warmup/cooldown blocks
remain unpaired. An invalid dependency or deadlock preserves the partial trace
and reports the blocked work instead of changing the schedule.

### V-shaped 1F1B (unsplit backward)

Select **1F1B-V (V-shape)** to run Appendix H's 1F1B-V from
[Pipeline Parallelism with Controllable Memory](https://arxiv.org/abs/2405.15362).
There are exactly two logical chunks per rank. For PP=4, the model's forward
path is `0 → 1 → 2 → 3 → 3 → 2 → 1 → 0`: rank 0 owns both the first and last
chunks and computes the loss. Same-device folds are local dependencies, not
P2P transfers. Model layer count must be divisible by `2 * PP`.

The repeating pattern and memory-constrained warmup/cooldown reordering follow
[`v1f1b.py` at upstream commit 39f2c186](https://github.com/sail-sg/zero-bubble-pipeline-parallelism/blob/39f2c186a27580b7051eb51e0f651e1efd0a8170/megatron/core/pipeline_parallel/zerobubble/scheduler/v1f1b.py).
Backward is a **single indivisible operation**, with the full configured
backward duration. Internal W markers only construct the reference order;
the simulator does not execute separate weight-gradient work.

This mode replays that fixed order through PipeScope's DES with nonblocking
sends, just-in-time receives, and a final send drain. The pattern was designed
for F:B=1:2; changing costs, sequence lengths, loss or P2P latency changes actual
runtime and memory without dynamically reordering the program. Interleaved
warmup formulas, group size, eager receive prefetch and MoE overlap do not apply.
This is not V-Half, V-Min, a Megatron transport reproduction, or an offload mode.

Activation peaks are measured from allocation/release events, including the
existing communication-buffer model; no `2/3` multiplier is applied. Compare
with ordinary 1F1B at the **same total layers**, per-layer costs, microbatch size,
sequence lengths and activation multiplier. A fixed `activationBytes` override
is per chunk and must be adjusted when changing the number of chunks. Static
baseline remains a user-supplied per-rank constant. Same-device input/output
storage aliasing and loss workspace remain outside the model.

`tests/fixtures/v-shaped-reference.json` records upstream operation orders for
PP=1,2,3,4,5,6,8; `scripts/generate-v-shaped-fixtures.py` regenerates it from a
local copy of the pinned source. Tests additionally verify dependency order,
FIFO matching, local folds, nonnegative/fully released memory, irregular token
lengths, positive loss and communication costs, and an event-derived memory
reduction at fixed total layers. Upstream redistribution terms are retained in
`public/THIRD_PARTY_NOTICES.txt` and included in the built site.
