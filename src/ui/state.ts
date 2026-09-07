import type { SimConfig, Trace } from '../sim/index.ts';

/** Shared horizontal time scale so the Gantt and memory panels stay aligned. */
export interface TimeScale {
  /** Pixels per time unit. */
  pxPerUnit: number;
  /** Time value shown at the left edge of the plot area. */
  offset: number;
}

export interface UiState {
  config: SimConfig;
  trace: Trace | null;
  error: string | null;
  /** Micro-batch pinned by click; null = none. */
  selectedMb: number | null;
  /** Op pinned by shift-click, shown with its critical-predecessor chain; null = none. */
  selectedOp: number | null;
  /** Transfer (by tag) pinned by clicking a wait or a lane bar; its bars on both ranks are highlighted. */
  selectedTransfer: string | null;
  /** Time under the pointer for the shared crosshair; null = none. */
  hoverTime: number | null;
  scale: TimeScale;
}

type Listener = (state: UiState) => void;

/** Minimal observable store; panels subscribe and re-render on change. */
export class Store {
  private state: UiState;
  private listeners = new Set<Listener>();

  constructor(initial: UiState) {
    this.state = initial;
  }

  get(): UiState {
    return this.state;
  }

  set(patch: Partial<UiState>): void {
    this.state = { ...this.state, ...patch };
    for (const l of this.listeners) l(this.state);
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    l(this.state);
    return () => this.listeners.delete(l);
  }

  /**
   * Subscribe, but only run `l` when one of `keys` changed (by reference).
   * Hover updates fire on every mouse move; panels that do not draw the
   * crosshair must not rebuild DOM or rewrite the URL on each of them.
   */
  subscribeTo(keys: (keyof UiState)[], l: Listener): () => void {
    let prev: UiState | null = null;
    return this.subscribe((s) => {
      if (prev && keys.every((k) => prev![k] === s[k])) return;
      prev = s;
      l(s);
    });
  }
}
