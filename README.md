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
