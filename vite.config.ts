import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// `base` is set for GitHub Pages project-site deployment (https://<user>.github.io/PipeScope/).
export default defineConfig({
  plugins: [react()],
  base: process.env.GITHUB_PAGES ? '/PipeScope/' : '/',
});
