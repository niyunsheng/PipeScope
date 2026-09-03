import { mountControls, recompute } from './ui/controls.ts';
import { mountGantt } from './ui/gantt.ts';
import { mountLengthsInfo } from './ui/lengthsInfo.ts';
import { t } from './ui/i18n.ts';
import { mountMemory } from './ui/memory.ts';
import { mountMetrics } from './ui/metrics.ts';
import { Store } from './ui/state.ts';
import { readUrl, writeUrl } from './ui/url.ts';

const initial = readUrl({
  schedule: 'interleaved-1f1b',
  pp: 4,
  vpp: 2,
  microBatches: 8,
  forwardTime: 1,
  backwardTime: 2,
  p2pLatency: 0,
  commModel: 'async',
  seqLen: 4096,
  hiddenSize: 4096,
  microBatchSize: 1,
  dtypeBytes: 2,
  activationMultiplier: 17,
  layersPerChunk: 2,
  linearAttnRatio: 6,
  lengthMode: 'uniform',
  lengthCv: 0.05,
  lengthSeed: 1,
  lengthOrder: 'asis',
});

const store = new Store({
  config: initial.config,
  trace: null,
  error: null,
  selectedMb: null,
  hoverMb: null,
  hoverTime: null,
  scale: { pxPerUnit: 10, offset: 0 },
});

const app = document.getElementById('root')!;
app.innerHTML = `
  <header class="topbar">
    <h1>PipeScope</h1>
    <span class="subtitle">${t('subtitle')}</span>
    <span id="schedule-slot" class="segmented"></span>
    <div class="topbar-actions">
      <button id="share" class="btn" type="button">${t('share')}</button>
      <a class="btn" href="https://github.com/niyunsheng/PipeScope" target="_blank" rel="noopener">${t('star')}</a>
    </div>
  </header>
  <div class="layout">
    <aside class="sidebar"><div id="controls"></div></aside>
    <main class="content">
      <section class="panel">
        <h2>${t('timeline')} <span class="hint-inline">${t('timelineHint')}</span></h2>
        <div id="lengths-info"></div>
        <div id="gantt" class="canvas-wrap"></div>
      </section>
      <section class="panel">
        <h2>${t('memoryPanel')}</h2>
        <div id="memory" class="canvas-wrap"></div>
      </section>
      <section class="panel" id="metrics"></section>
    </main>
  </div>
`;

// Share: copy the current URL (config and selection are in it).
const shareBtn = document.getElementById('share') as HTMLButtonElement;
shareBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(location.href);
    const label = shareBtn.textContent;
    shareBtn.textContent = t('copied');
    setTimeout(() => (shareBtn.textContent = label), 1500);
  } catch {
    window.prompt('URL', location.href);
  }
});

mountControls(document.getElementById('controls')!, document.getElementById('schedule-slot')!, store);
mountLengthsInfo(document.getElementById('lengths-info')!, store);
mountGantt(document.getElementById('gantt')!, store);
mountMemory(document.getElementById('memory')!, store);
mountMetrics(document.getElementById('metrics')!, store);
recompute(store);
if (initial.selectedMb !== null) store.set({ selectedMb: initial.selectedMb });
store.subscribe((s) => writeUrl(s.config, s.selectedMb));
