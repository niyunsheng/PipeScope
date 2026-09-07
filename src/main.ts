import { DEFAULT_CONFIG, validateConfig } from './sim/index.ts';
import { mountControls, recompute } from './ui/controls.ts';
import { mountGantt } from './ui/gantt.ts';
import { mountLengthsInfo } from './ui/lengthsInfo.ts';
import { t } from './ui/i18n.ts';
import { mountMemory } from './ui/memory.ts';
import { mountMetrics } from './ui/metrics.ts';
import { Store } from './ui/state.ts';
import { readUrl, writeUrl } from './ui/url.ts';


// A query string that does not describe a valid configuration (stale link,
// hand-edited values) is dropped entirely and the default view is shown.
let initial = readUrl(DEFAULT_CONFIG);
// Custom token lists are cycled to the micro-batch count later, so their length is not checked here.
if (validateConfig({ ...initial.config, tokens: undefined }).length > 0) {
  history.replaceState(null, '', location.pathname);
  initial = { config: { ...DEFAULT_CONFIG }, selectedMb: null };
}

const store = new Store({
  config: initial.config,
  trace: null,
  error: null,
  selectedMb: null,
  selectedOp: null,
  selectedTransfer: null,
  hoverTime: null,
  scale: { pxPerUnit: 10, offset: 0 },
});

const app = document.getElementById('root')!;
app.innerHTML = `
  <header class="topbar">
    <h1><a href="./" title="${t('homeHint')}">PipeScope</a></h1>
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
// history.replaceState is rate-limited by browsers; only touch it when the shareable state changes.
store.subscribeTo(['config', 'selectedMb'], (s) => writeUrl(s.config, DEFAULT_CONFIG, s.selectedMb));
