import { t } from './i18n.ts';
import { summarize } from './lengths.ts';
import type { Store } from './state.ts';

/** One-line token statistics under the timeline header, with the full list on demand. */
export function mountLengthsInfo(root: HTMLElement, store: Store): void {
  const details = document.createElement('details');
  details.className = 'lengths-info';
  details.open = true;
  const summary = document.createElement('summary');
  const list = document.createElement('p');
  list.className = 'hint token-list';
  details.appendChild(summary);
  details.appendChild(list);
  root.appendChild(details);
  store.subscribe((s) => {
    const toks = s.config.tokens;
    const show = (s.config.lengthMode ?? 'uniform') !== 'uniform' && toks && toks.length > 0;
    details.hidden = !show;
    if (!show || !toks) return;
    const sm = summarize(toks);
    summary.textContent = `${t('lengthList')}: ${t('lengthSummary', { min: sm.min, max: sm.max, mean: Math.round(sm.mean), std: Math.round(sm.std), cv: sm.cv.toFixed(2) })}`;
    list.textContent = toks.join(', ');
  });
}
