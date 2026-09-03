/**
 * Zero-dependency build: strip TypeScript types with Node's built-in
 * `stripTypeScriptTypes`, rewrite `.ts` import specifiers to `.js`, and copy
 * static assets into `dist/`. The output is plain ES modules served as-is,
 * which is all this app needs (no JSX, no bundler-only features).
 *
 *   node --experimental-strip-types scripts/build.ts [--base /PipeScope/]
 */
import { cpSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { dirname, join, relative } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const src = join(root, 'src');
const dist = join(root, 'dist');
const baseIdx = process.argv.indexOf('--base');
const base = baseIdx >= 0 ? process.argv[baseIdx + 1] : './';

rmSync(dist, { recursive: true, force: true });
mkdirSync(dist, { recursive: true });

function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : [p];
  });
}

let count = 0;
for (const file of walk(src)) {
  const rel = relative(root, file);
  if (file.endsWith('.ts')) {
    const code = readFileSync(file, 'utf8');
    const js = stripTypeScriptTypes(code, { mode: 'strip' }).replace(
      /(from\s+['"])(\.{1,2}\/[^'"]+)\.ts(['"])/g,
      '$1$2.js$3',
    );
    const out = join(dist, rel.replace(/\.ts$/, '.js'));
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, js);
    count++;
  } else {
    const out = join(dist, rel);
    mkdirSync(dirname(out), { recursive: true });
    cpSync(file, out);
  }
}

const html = readFileSync(join(root, 'index.html'), 'utf8')
  .replace('href="/src/styles.css"', `href="${base}src/styles.css"`)
  .replace('src="/src/main.ts"', `src="${base}src/main.js"`);
writeFileSync(join(dist, 'index.html'), html);
writeFileSync(join(dist, '.nojekyll'), '');
console.log(`built ${count} modules into dist/ (base=${base})`);
