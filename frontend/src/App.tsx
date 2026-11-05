import { useEffect, useRef, useState } from 'react'
import { errorMessage, getQuestion, searchQuestions } from './api'
import { QueryPanel, validTopK } from './components/QueryPanel'
import { ResultsPanel } from './components/ResultsPanel'
import { DetailPanel } from './components/DetailPanel'
import type { DetailState, SearchResult, SearchStatus } from './types'

const expiredMessage = '这道题已更新，原答案已隐藏。请刷新详情或重新搜索。'

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const fileRef = useRef<File | null>(null)
  const [ready, setReady] = useState(false)
  const [topK, setTopK] = useState('')
  const [error, setError] = useState('')
  const [status, setStatus] = useState<SearchStatus>('idle')
  const [results, setResults] = useState<SearchResult[]>([])
  const [selected, setSelected] = useState<SearchResult>()
  const selectedRef = useRef<SearchResult | undefined>(undefined)
  const [detail, setDetail] = useState<DetailState>({ status: 'idle' })
  const [staleIds, setStaleIds] = useState(new Set<string>())
  const expiredPreviews = useRef(new Set<string>())
  const searchVersion = useRef(0)
  const detailVersion = useRef(0)
  const searchRequest = useRef<AbortController | undefined>(undefined)
  const detailRequest = useRef<AbortController | undefined>(undefined)

  useEffect(() => () => {
    searchVersion.current++
    detailVersion.current++
    searchRequest.current?.abort()
    detailRequest.current?.abort()
  }, [])

  function chooseFile(next: File | null) {
    searchVersion.current++
    detailVersion.current++
    searchRequest.current?.abort()
    detailRequest.current?.abort()
    const valid = next && next.size > 0 ? next : null
    fileRef.current = valid
    setFile(valid)
    setReady(false)
    setError(next && !valid ? '图片为空，请重新选择或粘贴截图。' : '')
    setStatus('idle')
    setResults([])
    setSelected(undefined)
    selectedRef.current = undefined
    setDetail({ status: 'idle' })
    setStaleIds(new Set())
    expiredPreviews.current.clear()
  }

  async function selectResult(hit: SearchResult) {
    const version = ++detailVersion.current
    const queryVersion = searchVersion.current
    detailRequest.current?.abort()
    const controller = new AbortController()
    detailRequest.current = controller
    setSelected(hit)
    selectedRef.current = hit
    setDetail({ status: 'loading' })
    try {
      const question = await getQuestion(hit.question_id, controller.signal)
      if (version !== detailVersion.current || queryVersion !== searchVersion.current) return
      const updated = question.record_revision !== hit.record_revision
      if (updated) setStaleIds(previous => new Set(previous).add(hit.question_id))
      setDetail(expiredPreviews.current.has(JSON.stringify([question.question_id, question.record_revision]))
        ? { status: 'expired', message: expiredMessage } : { status: 'ready', question, updated })
    } catch (reason) {
      if (version === detailVersion.current && queryVersion === searchVersion.current && !controller.signal.aborted) {
        setDetail({ status: 'error', message: errorMessage(reason, 'detail') })
      }
    }
  }

  function expire(id: string, revision: string) {
    expiredPreviews.current.add(JSON.stringify([id, revision]))
    setStaleIds(previous => new Set(previous).add(id))
    if (selectedRef.current?.question_id === id) {
      setDetail(previous => previous.status === 'ready' && previous.question.record_revision !== revision
        ? previous : { status: 'expired', message: expiredMessage })
    }
  }

  async function search() {
    if (!file || !ready || !validTopK(topK) || status === 'loading') return
    const version = ++searchVersion.current
    detailVersion.current++
    searchRequest.current?.abort()
    detailRequest.current?.abort()
    const controller = new AbortController()
    searchRequest.current = controller
    setStatus('loading')
    setError('')
    setResults([])
    setSelected(undefined)
    selectedRef.current = undefined
    setDetail({ status: 'idle' })
    setStaleIds(new Set())
    expiredPreviews.current.clear()
    try {
      const hits = await searchQuestions(file, topK, controller.signal)
      if (version !== searchVersion.current) return
      setResults(hits)
      setStatus(hits.length ? 'success' : 'empty')
      if (hits[0]) void selectResult(hits[0])
    } catch (reason) {
      if (version === searchVersion.current && !controller.signal.aborted) {
        setError(errorMessage(reason, 'search'))
        setStatus('error')
      }
    }
  }

  return <div className="min-h-screen bg-[#f7f8fa] text-slate-800">
    <header className="border-b border-slate-200/80 bg-white">
      <div className="mx-auto flex max-w-[1600px] items-center gap-4 px-5 py-4 sm:px-8">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-white" aria-hidden="true">
          <svg width="23" height="23" viewBox="0 0 24 24" fill="none"><path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4z" fill="currentColor" /><path d="m14 17 2 2 4-5" stroke="#93c5fd" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" /></svg>
        </div>
        <span className="text-base font-bold tracking-tight">Img2Answer</span>
        <span className="mx-2 hidden h-5 w-px bg-slate-200 sm:block" />
        <span className="hidden text-sm text-slate-500 sm:block">以图找题</span>
        <span className="ml-auto inline-flex items-center gap-2 rounded-full border border-slate-200 px-3 py-1 text-xs text-slate-500"><span className="h-1.5 w-1.5 rounded-full bg-slate-400" />本地题库</span>
      </div>
    </header>
    <main className="mx-auto max-w-[1600px] px-4 py-7 sm:px-8 sm:py-9">
      <div className="mb-7 flex flex-wrap items-end justify-between gap-3">
        <div><p className="mb-2 text-[11px] font-semibold tracking-[0.16em] text-slate-400">题图检索工作台</p><h1 className="text-2xl font-bold tracking-tight sm:text-3xl">以图找题</h1><p className="mt-2 text-sm leading-6 text-slate-500">上传题目截图，查找相似题目，核对材料中的答案。</p></div>
        <p className="text-xs text-slate-400">上传题图 <span className="px-2 text-slate-300">/</span> 查找相似 <span className="px-2 text-slate-300">/</span> 核对原文</p>
      </div>
      <div className="grid items-start gap-5 lg:grid-cols-[270px_minmax(250px,0.85fr)_minmax(0,1.15fr)]">
        <QueryPanel file={file} topK={topK} ready={ready} loading={status === 'loading'} error={error}
          onFile={chooseFile} onTopK={setTopK} onSubmit={() => void search()} onReady={(imageFile, valid) => {
            if (imageFile !== fileRef.current) return
            setReady(valid)
            if (!valid) setError('无法读取这张图片，请重新选择有效的题目截图。')
          }} />
        <ResultsPanel results={results} status={status} selected={selected?.question_id} staleIds={staleIds}
          onSelect={hit => void selectResult(hit)} onExpired={expire} />
        <DetailPanel state={detail} onRefresh={() => { if (selected) void selectResult(selected) }}
          onExpired={revision => { if (selected) expire(selected.question_id, revision) }} />
      </div>
      <footer className="mt-7 flex flex-wrap justify-between gap-2 border-t border-slate-200/70 pt-4 text-[11px] leading-5 text-slate-400"><span>Img2Answer · 本地题库检索</span><span>相似结果供查找参考，自动提取答案需人工核对。</span></footer>
    </main>
  </div>
}
