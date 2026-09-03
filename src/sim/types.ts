/**
 * Core data types shared by the simulator and the UI.
 *
 * The simulator is a process-oriented discrete-event simulation (DES):
 * every pipeline rank is a sequential process executing a fixed `Program`
 * of compute and communication steps. Communication steps block until the
 * peer reaches its matching step (rendezvous) plus the wire latency, which
 * mirrors Megatron-LM's `batch_isend_irecv(...)` + `wait()` semantics.
 */

export type ScheduleName = 'gpipe' | '1f1b' | 'interleaved-1f1b';

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
export type OpKind = 'F' | 'B';

export interface SimConfig {
  schedule: ScheduleName;
  /** Number of pipeline ranks (devices). */
  pp: number;
  /** Number of virtual pipeline chunks per rank (1 = non-interleaved). */
  vpp: number;
  /** Number of micro-batches per global batch. */
  microBatches: number;
  /** Forward time of one micro-batch through one chunk. */
  forwardTime: number;
  /** Backward time of one micro-batch through one chunk. */
  backwardTime: number;
  /** Point-to-point latency of one activation / gradient transfer. */
  p2pLatency: number;
  /** Communication completion semantics; default `async`. */
  commModel?: CommModel;
  /**
   * Override: activation memory retained per (micro-batch, chunk) between
   * its forward and backward pass, in bytes, as a single number. When unset
   * the activation is derived from the model shape below as
   * `input + layersPerChunk * activationMultiplier * input`, where
   * `input = seqLen * microBatchSize * hiddenSize * dtypeBytes`.
   */
  activationBytes?: number;
  /** Sequence length per sample; default 4096. */
  seqLen?: number;
  /** Hidden size; default 4096. */
  hiddenSize?: number;
  /** Samples per micro-batch; default 1. */
  microBatchSize?: number;
  /** Bytes per activation element; default 2 (bf16). */
  dtypeBytes?: number;
  /**
   * Intermediate activation of one transformer layer as a multiple of the
   * layer input. Megatron's activation-recomputation paper gives
   * 34*s*b*h + 5*a*s^2*b bytes per layer vs. 2*s*b*h for the input, i.e.
   * 17 + 2.5*a*s/h; with flash attention (assumed) the attention-score term
   * vanishes, leaving 17 (the default).
   */
  activationMultiplier?: number;
  /** Total transformer layers in the model; layers per chunk = numLayers / (pp * vpp). Default 16. */
  numLayers?: number;
  /** Override for layers per virtual chunk (otherwise derived from numLayers). */
  layersPerChunk?: number;
  /** Static memory per rank in bytes (weights, grads, optimizer state); default 0. */
  baselineBytes?: number;
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
  kind: OpKind;
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
}

export type IdleReason = 'wait-recv' | 'wait-send';

/** A period during which a rank is blocked in a communication step. */
export interface IdleInterval {
  rank: number;
  start: number;
  end: number;
  reason: IdleReason;
  /** Human-readable description of the transfer that finished last. */
  detail: string;
  /** Time spent waiting for the peer to arrive at the matching step. */
  peerWait: number;
  /** Time spent on the wire after both peers arrived. */
  transfer: number;
}

export interface MemorySample {
  t: number;
  bytes: number;
  /** What changed at this event: input landed / forward started / backward ended. */
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

export interface Trace {
  config: SimConfig;
  ops: Op[];
  idles: IdleInterval[];
  /** memory[rank] is a step function sampled at every allocation / release. */
  memory: MemorySample[][];
  metrics: Metrics;
}
