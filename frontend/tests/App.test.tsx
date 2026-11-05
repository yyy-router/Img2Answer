import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'
import { getQuestion } from '../src/api'
import { hit, question } from './fixtures'
import type { Question } from '../src/types'

const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), {
  status, headers: { 'content-type': 'application/json' },
})
const imageResponse = () => new Response('image bytes', { headers: { 'content-type': 'image/png' } })
let fetchMock: ReturnType<typeof vi.fn<typeof fetch>>
let createUrl: ReturnType<typeof vi.fn>
let revokeUrl: ReturnType<typeof vi.fn>
let details: Question[]

beforeEach(() => {
  details = [question(1), question(2)]
  createUrl = vi.fn(() => `blob:test-${createUrl.mock.calls.length}`)
  revokeUrl = vi.fn()
  vi.stubGlobal('URL', Object.assign(class extends URL {}, { createObjectURL: createUrl, revokeObjectURL: revokeUrl }))
  fetchMock = vi.fn(async (input: string | URL | Request) => {
    const url = String(input)
    if (url === '/search/question') return json({ results: [hit(1), hit(2)] })
    if (url.startsWith('/questions/')) return json(details.find(item => item.question_id === decodeURIComponent(url.slice('/questions/'.length))))
    return imageResponse()
  })
  vi.stubGlobal('fetch', fetchMock)
})

async function upload(name = 'query.png') {
  fireEvent.change(screen.getByLabelText('题目图片'), { target: { files: [new File(['picture'], name, { type: 'image/png' })] } })
  fireEvent.load(await screen.findByAltText('查询图片预览'))
}

async function search() {
  await userEvent.click(screen.getByRole('button', { name: /搜索相似题/ }))
}

const detailPanel = () => screen.getByRole('region', { name: '题目详情' })

describe('query and results', () => {
  it('starts without sample results or an enabled search button', () => {
    render(<App />)
    expect(screen.getByRole('button', { name: /搜索相似题/ })).toBeDisabled()
    expect(screen.getByText('相似题目会出现在这里')).toBeVisible()
    expect(screen.getByText('选择一张题图，开始这次查找。')).toBeVisible()
    expect(screen.queryByText('材料中的答案')).not.toBeInTheDocument()
  })

  it('uploads a file, preserves server-default top K and shows current details', async () => {
    render(<App />)
    await upload()
    await search()
    expect(await within(detailPanel()).findByText('第 1 题的解析内容。')).toBeVisible()
    const [, options] = fetchMock.mock.calls.find(([url]) => url === '/search/question')!
    expect(options?.body).toBeInstanceOf(FormData)
    expect((options?.body as FormData).get('file')).toBeInstanceOf(File)
    expect((options?.body as FormData).has('top_k')).toBe(false)
    expect(options?.headers).toBeUndefined()
    expect(within(detailPanel()).getAllByText('待核对').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /查看第 1 题/ }))
      .toHaveAccessibleName('查看第 1 题：示例第 1 题：选择符合规律的图形。（待核对）')
  })

  it('validates the result count and submits an explicit integer', async () => {
    render(<App />)
    await upload()
    fireEvent.change(screen.getByLabelText(/结果数量/), { target: { value: '0' } })
    expect(screen.getByText('请输入大于 0 的整数。')).toBeVisible()
    expect(screen.getByRole('button', { name: /搜索相似题/ })).toBeDisabled()
    fireEvent.change(screen.getByLabelText(/结果数量/), { target: { value: '3' } })
    await search()
    const [, options] = fetchMock.mock.calls.find(([url]) => url === '/search/question')!
    expect((options?.body as FormData).get('top_k')).toBe('3')
  })

  it('accepts pasted images and leaves plain text paste alone', async () => {
    render(<App />)
    const zone = screen.getByRole('group', { name: '截图粘贴区域' })
    fireEvent.paste(zone, { clipboardData: { files: [], items: [] } })
    expect(screen.queryByAltText('查询图片预览')).not.toBeInTheDocument()
    fireEvent.paste(zone, { clipboardData: { files: [new File(['pasted'], 'clipboard.png', { type: 'image/png' })], items: [] } })
    fireEvent.load(await screen.findByAltText('查询图片预览'))
    expect(screen.getByText('clipboard.png')).toBeVisible()
    await search()
    expect(await screen.findByText('2 条结果')).toBeVisible()
  })

  it('rejects an empty file and an unreadable image', async () => {
    render(<App />)
    fireEvent.change(screen.getByLabelText('题目图片'), { target: { files: [new File([], 'empty.png')] } })
    expect(screen.getByRole('alert')).toHaveTextContent('图片为空')
    await upload()
    fireEvent.error(screen.getByAltText('查询图片预览'))
    expect(screen.getByRole('alert')).toHaveTextContent('无法读取这张图片')
    expect(screen.getByRole('button', { name: /搜索相似题/ })).toBeDisabled()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('shows an honest empty state', async () => {
    fetchMock.mockImplementation(async () => json({ results: [] }))
    render(<App />)
    await upload()
    await search()
    expect(await screen.findByText('没有找到有效的相似题')).toBeVisible()
    expect(screen.getByText('试试更完整的截图，或确认题库已建立索引。')).toBeVisible()
    expect(screen.queryByText('材料中的答案')).not.toBeInTheDocument()
  })

  it('explains a pending search without asking for another image', async () => {
    let finish!: (response: Response) => void
    fetchMock.mockImplementationOnce(() => new Promise(resolve => { finish = resolve }))
    render(<App />)
    await upload()
    await search()
    const results = screen.getByRole('region', { name: '相似题目' })
    expect(within(results).getByText('正在题库中寻找相似题…')).toBeVisible()
    expect(within(results).getByText('请稍候，正在比对题库。')).toBeVisible()
    expect(within(results).queryByText('选择一张题图，开始这次查找。')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /正在搜索/ })).toBeDisabled()
    await act(async () => { finish(json({ results: [] })) })
    expect(within(results).getByText('没有找到有效的相似题')).toBeVisible()
  })

  it.each([
    [400, '请检查图片内容和结果数量'], [413, '图片超过当前服务允许的大小'],
    [503, '题库或索引尚未就绪'], [500, '服务暂时无法完成请求'],
  ])('explains HTTP %i without displaying backend details', async (status, message) => {
    fetchMock.mockImplementation(async () => json({ detail: 'private backend path /secrets' }, Number(status)))
    render(<App />)
    await upload()
    await search()
    expect(await screen.findByRole('alert')).toHaveTextContent(String(message))
    expect(screen.queryByText(/private backend/)).not.toBeInTheDocument()
    expect(screen.getByText('query.png')).toBeVisible()
    const results = screen.getByRole('region', { name: '相似题目' })
    expect(within(results).getByText('请查看错误提示，处理后重新搜索。')).toBeVisible()
    expect(within(results).queryByText('选择一张题图，开始这次查找。')).not.toBeInTheDocument()
    expect(screen.getByRole('button', { name: /搜索相似题/ })).toBeEnabled()
  })

  it('allows retry after a network failure', async () => {
    const normal = fetchMock.getMockImplementation()!
    fetchMock.mockRejectedValueOnce(new TypeError('network details'))
    render(<App />)
    await upload()
    await search()
    expect(await screen.findByRole('alert')).toHaveTextContent('请确认本地服务已启动')
    fetchMock.mockImplementation(normal)
    await search()
    expect(await screen.findByText('2 条结果')).toBeVisible()
  })
})

describe('question integrity and resource lifecycle', () => {
  it('distinguishes missing and conflicting answers', async () => {
    details = [question(1, { answer: null, review_reasons: ['missing_answer'] }),
      question(2, { answer: null, review_reasons: ['conflicting_answer'] })]
    render(<App />)
    await upload()
    await search()
    expect((await within(detailPanel()).findAllByText('未识别到答案'))[0]).toBeVisible()
    await userEvent.click(screen.getByRole('button', { name: /查看第 2 题/ }))
    expect(await within(detailPanel()).findByText('答案存在冲突，需核对')).toBeVisible()
  })

  it('renders OCR markup as text rather than executable HTML', async () => {
    details[0].stem_text = '<img src=x onerror=alert(1)>'
    render(<App />)
    await upload()
    await search()
    expect(await within(detailPanel()).findByText('<img src=x onerror=alert(1)>')).toBeVisible()
    expect(detailPanel().querySelector('img[src="x"]')).toBeNull()
  })

  it('does not attach a late search response to a new query image', async () => {
    const normal = fetchMock.getMockImplementation()!
    let resolveOld!: (response: Response) => void
    let requests = 0
    fetchMock.mockImplementation((input, init) => {
      if (input === '/search/question') {
        if (++requests === 1) return new Promise(resolve => { resolveOld = resolve })
        return Promise.resolve(json({ results: [hit(2)] }))
      }
      return normal(input, init)
    })
    render(<App />)
    await upload('old.png')
    await search()
    const oldSignal = fetchMock.mock.calls.find(([url]) => url === '/search/question')![1]?.signal
    await upload('new.png')
    expect(oldSignal?.aborted).toBe(true)
    await search()
    expect(await within(detailPanel()).findByText('第 2 题的解析内容。')).toBeVisible()
    await act(async () => { resolveOld(json({ results: [hit(1)] })) })
    expect(screen.queryByRole('button', { name: /查看第 1 题/ })).not.toBeInTheDocument()
    expect(within(detailPanel()).getByText('第 2 题的解析内容。')).toBeVisible()
  })

  it('does not overwrite the selected question with an older detail request', async () => {
    const normal = fetchMock.getMockImplementation()!
    let resolveOld!: (response: Response) => void
    fetchMock.mockImplementation((input, init) => {
      if (String(input) === `/questions/${encodeURIComponent(question(1).question_id)}`) {
        return new Promise(resolve => { resolveOld = resolve })
      }
      return normal(input, init)
    })
    render(<App />)
    await upload()
    await search()
    await userEvent.click(await screen.findByRole('button', { name: /查看第 2 题/ }))
    expect(await within(detailPanel()).findByText('第 2 题的解析内容。')).toBeVisible()
    await act(async () => { resolveOld(json(question(1))) })
    expect(within(detailPanel()).queryByText('第 1 题的解析内容。')).not.toBeInTheDocument()
  })

  it('hides expired answers after 409 and refreshes to the current revision', async () => {
    const normal = fetchMock.getMockImplementation()!
    fetchMock.mockImplementation((input, init) => String(input).startsWith('/image-content/image-1')
      ? Promise.resolve(json({}, 409)) : normal(input, init))
    render(<App />)
    await upload()
    await search()
    expect(await within(detailPanel()).findByRole('alert')).toHaveTextContent('原答案已隐藏')
    expect(screen.getByRole('button', { name: /查看第 1 题/ })).toHaveAccessibleName(/记录已更新/)
    expect(within(detailPanel()).queryByText('材料中的答案')).not.toBeInTheDocument()
    details[0] = question(1, { answer: 'B', record_revision: 'updated', image: {
      ...question(1).image, preview_url: '/image-content/current?revision=updated',
    } })
    await userEvent.click(within(detailPanel()).getByRole('button', { name: '刷新详情' }))
    expect(await within(detailPanel()).findByText('题目记录已更新。以下是当前详情，请重新搜索以确认匹配。')).toBeVisible()
    expect(within(detailPanel()).getByText('材料中的答案')).toBeVisible()
  })

  it('encodes question identifiers without altering their contents', async () => {
    const id = '中文/题库:第 1 题'
    fetchMock.mockResolvedValueOnce(json(question(1, { question_id: id })))
    await getQuestion(id, new AbortController().signal)
    expect(fetchMock.mock.calls[0][0]).toBe(`/questions/${encodeURIComponent(id)}`)
  })

  it('does not hide a refreshed detail when an older thumbnail later reports 409', async () => {
    const normal = fetchMock.getMockImplementation()!
    let finishOldThumbnail!: (response: Response) => void
    details[0] = question(1, { record_revision: 'current', image: {
      ...question(1).image, preview_url: '/image-content/current?revision=current',
    } })
    fetchMock.mockImplementation((input, init) => String(input).startsWith('/image-content/image-1')
      ? new Promise(resolve => { finishOldThumbnail = resolve }) : normal(input, init))
    render(<App />)
    await upload()
    await search()
    await within(detailPanel()).findByText('材料中的答案')
    await act(async () => { finishOldThumbnail(json({}, 409)) })
    expect(within(detailPanel()).getByText('材料中的答案')).toBeVisible()
    expect(within(detailPanel()).queryByText(/原答案已隐藏/)).not.toBeInTheDocument()
  })

  it('releases image object URLs and cancels requests on clear and unmount', async () => {
    const view = render(<App />)
    await upload()
    await search()
    await within(detailPanel()).findByAltText('完整题目截图')
    await userEvent.click(screen.getByRole('button', { name: '清空图片' }))
    expect(screen.queryByText('材料中的答案')).not.toBeInTheDocument()
    view.unmount()
    const created = createUrl.mock.results.map(result => result.value)
    expect(created.length).toBeGreaterThan(1)
    expect(new Set(revokeUrl.mock.calls.map(call => call[0]))).toEqual(new Set(created))
    expect(fetchMock.mock.calls.every(([, init]) => init?.signal?.aborted)).toBe(true)
  })

  it('does not create an object URL when an old image fetch resolves after unmount', async () => {
    const normal = fetchMock.getMockImplementation()!
    let finish!: (response: Response) => void
    fetchMock.mockImplementation((input, init) => String(input).startsWith('/image-content/')
      ? new Promise(resolve => { finish = resolve }) : normal(input, init))
    const view = render(<App />)
    await upload()
    await search()
    await within(detailPanel()).findByText('材料中的答案')
    const count = createUrl.mock.calls.length
    view.unmount()
    await act(async () => { finish(imageResponse()) })
    expect(createUrl).toHaveBeenCalledTimes(count)
  })
})
