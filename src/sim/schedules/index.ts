import { Topology } from './common.ts';
import { vShapedProgram } from './vShaped.ts';
import { combineMoe } from '../moe.ts';
import type { Program, ScheduleName, SimConfig } from '../types.ts';
import { gpipeProgram } from './gpipe.ts';
import { customProgram } from './custom.ts';
import { interleavedProgram } from './interleaved.ts';
import { oneF1BProgram } from './oneF1B.ts';

export type ScheduleGenerator = (cfg: SimConfig) => Program;

export interface ScheduleInfo {
  name: ScheduleName;
  label: string;
  supportsVpp: boolean;
  generate: ScheduleGenerator;
}

export const SCHEDULES: Record<ScheduleName, ScheduleInfo> = {
  gpipe: {
    name: 'gpipe',
    label: 'GPipe',
    supportsVpp: false,
    generate: gpipeProgram,
  },
  '1f1b': {
    name: '1f1b',
    label: '1F1B',
    supportsVpp: false,
    generate: oneF1BProgram,
  },
  'interleaved-1f1b': {
    name: 'interleaved-1f1b',
    label: 'Interleaved 1F1B (VPP)',
    supportsVpp: true,
    generate: interleavedProgram,
  },
  '1f1b-v': { name: '1f1b-v', label: '1F1B-V (V-shape)', supportsVpp: true, generate: vShapedProgram },
  custom: {
    name: 'custom',
    label: 'Custom',
    supportsVpp: true,
    generate: customProgram,
  },
};

export function buildProgram(cfg: SimConfig): Program {
  const info = SCHEDULES[cfg.schedule];
  if (!info) throw new Error(`Unknown schedule: ${cfg.schedule}`);
  const program = info.generate(cfg);
  return withLoss(cfg.moeOverlap ? combineMoe(program) : program, cfg);
}

/**
 * Insert the loss computation right after every forward on the last stage,
 * where Megatron computes it inside `forward_step`. Generators stay unaware
 * of it. The op is always present because it is a real dependency (the last
 * backward needs the loss, the loss needs the last forward); with
 * `lossTime = 0` it has no duration and the UI does not draw it.
 */
function withLoss(program: Program, cfg: SimConfig): Program {
  const topology = new Topology(cfg.pp, cfg.vpp, cfg.schedule === '1f1b-v');
  const lastStage = topology.numStages - 1;
  return program.map((steps, rank) =>
    steps.flatMap((step) =>
      step.type === 'compute' && step.kind === 'F' && topology.stage(step.chunk, rank) === lastStage
        ? [step, { type: 'compute' as const, kind: 'L' as const, mb: step.mb, chunk: step.chunk }]
        : [step],
    ),
  );
}
