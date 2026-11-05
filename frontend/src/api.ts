import type { Question, SearchResult } from './types'

export class ApiError extends Error {
  constructor(public status: number) { super(`Request failed (${status})`) }
}

export async function checkedFetch(url: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(url, init)
  if (!response.ok) throw new ApiError(response.status)
  return response
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function readQuestion(value: unknown): Question {
  if (!isRecord(value) || !['question_id', 'image_id', 'record_revision', 'document_id', 'section',
    'question_no', 'stem_text', 'analysis_text'].every(key => typeof value[key] === 'string') ||
    !Number.isInteger(value.page_number) || Number(value.page_number) < 1 ||
    !isRecord(value.options) || !Object.values(value.options).every(option => typeof option === 'string') ||
    ![null, 'A', 'B', 'C', 'D'].includes(value.answer as string | null) ||
    value.parse_status !== 'needs_review' || !Array.isArray(value.review_reasons) ||
    !value.review_reasons.every(reason => typeof reason === 'string') || !isRecord(value.image) ||
    typeof value.image.preview_url !== 'string' || !value.image.preview_url.startsWith('/image-content/')) {
    throw new Error('Invalid question response')
  }
  return value as unknown as Question
}

export async function searchQuestions(file: File, topK: string, signal: AbortSignal): Promise<SearchResult[]> {
  const form = new FormData()
  form.append('file', file)
  if (topK.trim()) form.append('top_k', topK.trim())
  const response = await checkedFetch('/search/question', { method: 'POST', body: form, signal })
  const value: unknown = await response.json()
  if (!isRecord(value) || !Array.isArray(value.results)) throw new Error('Invalid search response')
  return value.results.map(item => {
    readQuestion(item)
    if (!isRecord(item) || !Number.isInteger(item.rank) ||
      !(item.distance === null || typeof item.distance === 'number' && Number.isFinite(item.distance))) {
      throw new Error('Invalid search result')
    }
    return item as unknown as SearchResult
  })
}

export async function getQuestion(id: string, signal: AbortSignal): Promise<Question> {
  const response = await checkedFetch(`/questions/${encodeURIComponent(id)}`, { signal })
  return readQuestion(await response.json())
}

export function errorMessage(error: unknown, context: 'search' | 'detail' | 'image'): string {
  const status = error instanceof ApiError ? error.status : 0
  if (status === 409) return '这道题已更新，请刷新详情或重新搜索。'
  if (status === 404) return context === 'detail' ? '这道题已不可用，请选择其他结果或重新搜索。' : '题图暂不可用'
  if (status === 413) return '图片超过当前服务允许的大小，请缩小图片后重试。'
  if (status === 400 || status === 422) return '请检查图片内容和结果数量后重试。'
  if (status === 503) return '题库或索引尚未就绪，请按运行说明准备数据后重试。'
  if (context === 'image') return '题图加载失败'
  if (status >= 500) return '服务暂时无法完成请求，请稍后重试。'
  return '请求未完成，请确认本地服务已启动后重试。'
}
