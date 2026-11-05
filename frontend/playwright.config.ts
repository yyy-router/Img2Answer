import { defineConfig } from '@playwright/test'

const realUrl = process.env.IMG2ANSWER_E2E_REAL_URL

export default defineConfig({
  testDir: './e2e',
  workers: 1,
  timeout: 30_000,
  reporter: 'list',
  use: {
    baseURL: realUrl || 'http://127.0.0.1:5173',
    channel: process.env.PLAYWRIGHT_CHANNEL || (process.platform === 'win32' ? 'msedge' : undefined),
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  projects: [
    { name: 'desktop', use: { viewport: { width: 1440, height: 1000 } } },
    { name: 'narrow', use: { viewport: { width: 390, height: 844 } } },
  ],
  webServer: realUrl ? undefined : {
    command: 'npm run dev', url: 'http://127.0.0.1:5173', reuseExistingServer: !process.env.CI,
  },
})
