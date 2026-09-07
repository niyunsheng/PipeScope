/**
 * Core data types shared by the simulator and the UI.
 *
 * The simulator is a process-oriented discrete-event simulation (DES):
 * every pipeline rank is a sequential process executing a fixed `Program`
 * of compute and communication steps. Communication steps block until the
 * peer reaches its matching step (rendezvous) plus the wire latency, which
 * mirrors Megatron-LM's `batch_isend_irecv(...)` + `wait()` semantics.
 */

export type ScheduleName = 'gpipe' | '1f1b' | 'interleaved-1f1b' | 'custom';

/**
 * How a point-to-point transfer completes.
 * - `async`: buffered / asynchronous send. The sender never blocks; data
 *   lands at `send time + latency` and a recv completes at
 *   `max(recv posted, data landed)`. Close to `overlap_p2p_comm` or an
 *   isend whose wait is deferred. This is the default because it matches
 *   the intuition that data sent long ago is available immediately.
 * - `sync`: rendezvous. Data only moves once both peers have posted, so a
 *   transfer completes at `max(send posted, recv posted) + latency` and both
 *   ranks block until then. This mirrors NCCL send/recv kernels under
 *   Megatron's synchronous `batch_isend_irecv` + `wait` path.
 */
export type CommModel = 'async' | 'sync';

/** Forward or backward pass of one micro-batch through one model chunk. */
/** Forward, backward, or the loss computation that sits between them on the last stage. */
export type OpKind = 'F' | 'B' | 'L';
/** Kinds that flow between stages; the loss never leaves the last stage. */
export type TensorKind = 'F' | 'B';

export interface SimConfig {
  schedule: ScheduleName;
  /** Number of pipeline ranks (devices). */
  pp: number;
  /** Number of virtual pipeline chunks per rank (1 = non-interleaved). */
  vpp: number;
  /** Number of micro-batches per global batch. */
  microBatches: number;
  /**
   * Forward time of one micro-batch through one transformer layer. A chunk
   * costs `layersPerChunk * forwardTime`, so schedules with different chunk
   * counts (1F1B vs. VPP) stay comparable for the same model.
   */
  forwardTime: number;
  /** Backward time of one micro-batch through one transformer layer. */
  backwardTime: number;
  /**
   * Loss computation time on the last stage, between a micro-batch's last
   * forward and its first backward (Megatron runs it inside the last
   * stage's `forward_step`). The op always exists as a dependency; 0 = no
   * duration and not drawn. Memory is not modelled.
   */
  lossTime: number;
  /** Point-to-point latency of one activation / gradient transfer. */
  p2pLatency: number;
  /** Communication completion semantics. */
  commModel: CommModel;
  /**
   * `custom` schedule only. Warmup forwards per rank as a formula over
   * pp, vpp, m, r, G, total (see `sim/formula.ts`); Megatron's is
   * `2 * (pp - r - 1) + (vpp - 1) * G`.
   */
  warmupFormula: string;
  /**
   * Interleaved and custom schedules: micro-batches processed on one chunk
   * before switching (Megatron's `microbatch_group_size_per_vp_stage`).
   * Valid range is [pp, m] with m mod G either 0 or ≥ pp; Megatron defaults
   * to pp, and the UI keeps it equal to pp until it is edited by hand.
   */
  groupSize: number;
  /**
   * When a steady-state forward's output is sent: right after the forward
   * (`F`, Megatron 1F1B and the interleaved overlap path) or after the
   * backward together with the gradient (`B`, interleaved synchronous path).
   */
  sendAfter: 'F' | 'B';
  /**
   * When the gradient a backward needs is waited for: in the step that ends
   * the previous round, before the forward (`beforeF`, interleaved
   * synchronous path) or right before the backward itself (`beforeB`,
   * 1F1B and the interleaved overlap path). See `megatronPlacement`.
   */
  waitGrad: 'beforeF' | 'beforeB';
  /**
   * Override: activation memory retained per (micro-batch, chunk) between
   * its forward and backward pass, in bytes, as a single number. When unset
   * the activation is derived from the model shape below as
   * `input + layersPerChunk * activationMultiplier * input` (layersPerChunk = numLayers / (pp * vpp)), where
   * `input = seqLen * microBatchSize * hiddenSize * dtypeBytes`.
   */
  activationBytes?: number;
  /** Sequence length per sample. */
  seqLen: number;
  /** Hidden size. */
  hiddenSize: number;
  /** Samples per micro-batch. */
  microBatchSize: number;
  /** Bytes per activation element (2 = bf16). */
  dtypeBytes: number;
  /**
   * Intermediate activation of one transformer layer as a multiple of the
   * layer input. Megatron's activation-recomputation paper gives
   * 34*s*b*h + 5*a*s^2*b bytes per layer vs. 2*s*b*h for the input, i.e.
   * 17 + 2.5*a*s/h; with flash attention (assumed) the attention-score term
   * vanishes, leaving 17.
   */
  activationMultiplier: number;
  /** Total transformer layers in the model; must be divisible by pp * vpp (layers per chunk). */
  numLayers: number;
  /**
   * Ratio of the linear-layer FLOP coefficient to the core-attention FLOP
   * coefficient, k. Per layer, linear FLOPs ∝ k * s * h^2 and core-attention
   * FLOPs ∝ s^2 * h (both up to the same constant), so linear : attention =
   * k * h : s. GPT: 24 s h^2 vs 4 s^2 h -> k = 6. The quadratic
   * share at the reference length is a = s / (k * h + s) and compute of a
   * micro-batch with r = tokens / seqLen scales as (1 - a) * r + a * r^2.
   */
  linearAttnRatio: number;
  /** Static memory per rank in bytes (weights, grads, optimizer state). */
  baselineBytes: number;
  /**
   * Tokens per micro-batch (length = microBatches). When set, compute time
   * and activation memory of micro-batch i scale with tokens[i] / seqLen:
   * memory linearly, compute as (1 - a) * r + a * r^2 where a is the share of
   * attention FLOPs at the reference length (see `quadraticShare`).
   * Unset = every micro-batch has seqLen tokens.
   */
  tokens?: number[];
  /** How `tokens` were generated (UI / URL metadata; the simulator ignores these). */
  lengthMode: 'uniform' | 'lognormal' | 'custom';
  lengthCv: number;
  lengthSeed: number;
  lengthOrder: 'asis' | 'asc' | 'desc' | 'alternate';
}

/** A compute step: run F or B of `mb` through model chunk `chunk`. */
export interface ComputeStep {
  type: 'compute';
  kind: OpKind;
  mb: number;
  chunk: number;
}

/**
 * One direction of a point-to-point transfer.
 * `tag` identifies the logical tensor (e.g. `F:3:2` = forward output of
 * micro-batch 3 produced by global stage 2). Tags are asserted to match on
 * both ends so schedule bugs surface as errors instead of silent mismatches.
 */
export interface Transfer {
  kind: TensorKind;
  peer: number;
  tag: string;
  mb: number;
}

/**
 * A blocking, batched communication step. Megatron issues sends and recvs
 * together (`send_forward_recv_backward`, ...) and waits for all of them.
 */
export interface CommStep {
  type: 'comm';
  sends: Transfer[];
  recvs: Transfer[];
}

export type Step = ComputeStep | CommStep;

/** Per-rank step sequences. `program[rank]` is executed sequentially. */
export type Program = Step[][];

export type IdleReason = 'wait-recv' | 'wait-send';

/**
 * A binding predecessor of an op: something that finished exactly when the
 * op started, so delaying it would delay the op. `program`: the previous
 * compute op on the same rank. `wait-recv` / `wait-send`: a transfer in the
 * comm step before this op completed at the op's start, and `op` is the
 * peer's compute op that immediately preceded the peer posting its end.
 */
export interface BlockedBy {
  reason: 'program' | IdleReason;
  /** `Op.id` of the predecessor; ids index into `Trace.ops`. */
  op: number;
  /** For waits: tag of the transfer whose completion released this rank. */
  transferTag?: string;
}

/** A scheduled compute op with its simulated timing. */
export interface Op {
  id: number;
  rank: number;
  chunk: number;
  /** Global stage index = chunk * pp + rank (interleaved placement). */
  stage: number;
  mb: number;
  kind: OpKind;
  start: number;
  end: number;
  /**
   * All binding predecessors. With zero latency and integer durations ties
   * are common, so several constraints can bind at once. Ordered by priority:
   * cross-rank waits first (they carry pipeline information), then the
   * same-rank program order; the critical-path chain follows the first one.
   * Empty only for an op that starts at t = 0.
   */
  predecessors: BlockedBy[];
}

/** A period during which a rank is blocked in a communication step. */
export interface IdleInterval {
  rank: number;
  start: number;
  end: number;
  reason: IdleReason;
  /** Tag of the transfer that finished last, i.e. the one this rank was waiting on. */
  transferTag: string;
  /** Time spent waiting for the peer to post its end of the transfer. */
  peerWait: number;
  /**
   * Remaining idle time after the peer posted, i.e. wire time still in
   * flight. Always `peerWait + transfer === end - start`.
   */
  transfer: number;
}

/** One point-to-point transfer as it actually happened in the simulation. */
export interface TransferRecord {
  tag: string;
  kind: TensorKind;
  mb: number;
  from: number;
  to: number;
  /** When the sender posted its send. */
  sendPosted: number;
  /** When the receiver posted its recv. */
  recvPosted: number;
  /** When data started moving: `sendPosted` under async, `max(sendPosted, recvPosted)` under sync. */
  start: number;
  /** When data landed on the receiver: `start + wire`. */
  landed: number;
  /** Sender's compute op that produced the tensor (null if none preceded the send). */
  producer: number | null;
}

export interface MemorySample {
  t: number;
  bytes: number;
  /** What changed at this event: input buffer allocated / forward started / backward ended. */
  event: 'input' | 'forward' | 'release';
  /** Activations resident on the rank right after this event, as "mb:chunk". */
  resident: string[];
}

export interface RankMetrics {
  rank: number;
  busy: number;
  waitRecv: number;
  waitSend: number;
  /** Idle time after the rank finished its program, until the global end. */
  tail: number;
  utilization: number;
  peakMemory: number;
}

export interface Metrics {
  totalTime: number;
  /** Per-rank compute time if there were no bubbles at all. */
  idealTime: number;
  /** (totalTime - idealTime) / idealTime */
  overheadRatio: number;
  /** Fraction of (pp * totalTime) that is not compute. */
  bubbleFraction: number;
  ranks: RankMetrics[];
}

/**
 * Why a simulation stopped early. Everything simulated up to that point is
 * still returned, so the timeline can show what ran and mark what failed.
 */
export interface SimFailure {
  kind: 'program' | 'deadlock' | 'tag-mismatch';
  message: string;
  /** Ranks stuck in a comm step when the run stopped, and what they were waiting on. */
  blocked: { rank: number; since: number; waiting: string }[];
  /** For `program`: the compute step that could not legally run, placed where it would have started. */
  op: Op | null;
}

export interface Trace {
  config: SimConfig;
  ops: Op[];
  idles: IdleInterval[];
  transfers: TransferRecord[];
  /** Null when the whole program ran to completion. */
  failure: SimFailure | null;
  /** memory[rank] is a step function sampled at every allocation / release. */
  memory: MemorySample[][];
  metrics: Metrics;
}
