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
  /** Micro-batch under the pointer; null = none. */
  hoverMb: number | null;
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
}

/** The micro-batch currently emphasised: pinned selection wins over hover. */
export function activeMb(s: UiState): number | null {
  return s.selectedMb ?? s.hoverMb;
}
