# PipeScope

**Live demo: <https://niyunsheng.github.io/PipeScope/>**

Pipeline-parallel (PP) schedule visualizer built on a discrete-event simulator. Ranks run their Megatron-style programs step by step and communication blocks exactly where it would in practice, so timelines reflect real behaviour under p2p latency, sync vs. async communication, and other scenarios instead of idealized diagrams.

Shows GPipe / 1F1B / interleaved 1F1B (VPP) timelines per rank and chunk, the full path of any micro-batch, per-rank activation memory, and bubble attribution.

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
