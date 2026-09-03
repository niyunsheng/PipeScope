import type { Program, ScheduleName, SimConfig } from '../types.ts';
import { gpipeProgram } from './gpipe.ts';
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
};

export function buildProgram(cfg: SimConfig): Program {
  const info = SCHEDULES[cfg.schedule];
  if (!info) throw new Error(`Unknown schedule: ${cfg.schedule}`);
  return info.generate(cfg);
}
