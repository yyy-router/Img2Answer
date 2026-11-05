import { test, expect } from '@playwright/test'
import { hit, question, pngBase64 } from '../tests/fixtures'

const picture = Buffer.from(pngBase64, 'base64')
const query = { name: 'query.png', mimeType: 'image/png', buffer: picture }

test.describe('browser interaction with generated API responses', () => {
  test.skip(Boolean(process.env.IMG2ANSWER_E2E_REAL_URL), 'Real API smoke is verified separately')
  test.beforeEach(async ({ page }) => {
    await page.route('**/search/question', route => route.fulfill({ json: { results: [hit(1), hit(2)] } }))
    await page.route('**/questions/**', route => {
      const id = decodeURIComponent(new URL(route.request().url()).pathname.slice('/questions/'.length))
      return route.fulfill({ json: id === question(2).question_id ? question(2, {
        answer: null, review_reasons: ['missing_answer'],
      }) : question(1) })
    })
    await page.route('**/image-content/**', route => route.fulfill({ body: picture, contentType: 'image/png' }))
    await page.goto('/')
  })

  test('uploads, searches, changes selected question and previews source images', async ({ page }, info) => {
    const errors: string[] = []
    page.on('pageerror', error => errors.push(error.message))
    await expect(page.getByRole('button', { name: '搜索相似题' })).toBeDisabled()
    await page.screenshot({ path: info.outputPath('initial.png'), fullPage: true })
    await page.getByLabel('题目图片').setInputFiles(query)
    await expect(page.getByRole('button', { name: '搜索相似题' })).toBeEnabled()
    await page.getByRole('button', { name: '搜索相似题' }).click()
    const detail = page.getByRole('region', { name: '题目详情' })
    await expect(detail.getByText('第 1 题的解析内容。')).toBeVisible()
    await expect(detail.getByAltText('完整题目截图')).toBeVisible()
    await expect(detail.getByAltText('完整题目截图')).toHaveJSProperty('naturalWidth', 1)
    await expect(detail.getByRole('link', { name: /查看原图/ })).toHaveAttribute('href', /^blob:/)
    await page.screenshot({ path: info.outputPath('results.png'), fullPage: true })
    await page.getByRole('button', { name: /查看第 2 题/ }).click()
    await expect(detail.getByText('未识别到答案', { exact: true }).first()).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    expect(errors).toEqual([])
  })

  test('accepts an image paste without requiring clipboard permissions', async ({ page }) => {
    await page.getByRole('group', { name: '截图粘贴区域' }).evaluate((zone, base64) => {
      const bytes = Uint8Array.from(atob(base64), char => char.charCodeAt(0))
      const transfer = new DataTransfer()
      transfer.items.add(new File([bytes], 'pasted.png', { type: 'image/png' }))
      zone.dispatchEvent(new ClipboardEvent('paste', { clipboardData: transfer, bubbles: true }))
    }, pngBase64)
    await expect(page.getByText('pasted.png', { exact: true })).toBeVisible()
    await expect(page.getByRole('button', { name: '搜索相似题' })).toBeEnabled()
    await page.getByRole('button', { name: '搜索相似题' }).click()
    await expect(page.getByText('2 条结果')).toBeVisible()
  })

  test('reports empty results and a service-unavailable response with retry', async ({ page }) => {
    await page.route('**/search/question', route => route.fulfill({ status: 503, json: { detail: 'private' } }))
    await page.getByLabel('题目图片').setInputFiles(query)
    await page.getByRole('button', { name: '搜索相似题' }).click()
    await expect(page.getByRole('alert')).toContainText('题库或索引尚未就绪')
    await page.route('**/search/question', route => route.fulfill({ json: { results: [] } }))
    await page.getByRole('button', { name: '搜索相似题' }).click()
    await expect(page.getByText('没有找到有效的相似题')).toBeVisible()
  })

  test('has keyboard-visible focus and validates the optional count', async ({ page }) => {
    await page.keyboard.press('Tab')
    await expect(page.getByRole('group', { name: '截图粘贴区域' })).toBeFocused()
    await page.getByLabel('题目图片').setInputFiles(query)
    await page.getByLabel(/结果数量/).fill('0')
    await expect(page.getByText('请输入大于 0 的整数。')).toBeVisible()
    await expect(page.getByRole('button', { name: '搜索相似题' })).toBeDisabled()
  })
})

test.describe('built UI with real FastAPI and Chroma', () => {
  test.skip(!process.env.IMG2ANSWER_E2E_REAL_URL, 'Requires the generated local fixture server')
  test('uploads a generated question, previews the real crop and preserves missing-answer state', async ({ page }, info) => {
    await page.goto('/')
    await page.getByLabel('题目图片').setInputFiles(process.env.IMG2ANSWER_E2E_QUERY_IMAGE!)
    await page.getByLabel(/结果数量/).fill('2')
    await page.getByRole('button', { name: '搜索相似题' }).click()
    const detail = page.getByRole('region', { name: '题目详情' })
    await expect(detail.getByText('材料中的答案')).toBeVisible()
    await expect(detail.getByAltText('完整题目截图')).toBeVisible()
    expect(await detail.getByAltText('完整题目截图').evaluate(image => (image as HTMLImageElement).naturalWidth)).toBeGreaterThan(1)
    await page.screenshot({ path: info.outputPath('real-api.png'), fullPage: true })
    await page.getByRole('button', { name: /查看第 2 题/ }).click()
    await expect(detail.getByText('未识别到答案', { exact: true }).first()).toBeVisible()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  })
})
