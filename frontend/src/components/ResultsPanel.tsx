import type { SearchResult, SearchStatus } from '../types'
import { Icon } from './Icon'
import { RemoteImage } from './RemoteImage'

export function ResultsPanel({ results, status, selected, staleIds, onSelect, onExpired }: {
  results: SearchResult[]; status: SearchStatus; selected?: string; staleIds: Set<string>
  onSelect: (result: SearchResult) => void; onExpired: (id: string, revision: string) => void
}) {
  return <section className="panel min-w-0" aria-labelledby="results-heading" aria-busy={status === 'loading'}>
    <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-4">
      <span className="step-label">02</span><h2 id="results-heading" className="font-semibold">相似题目</h2>
      {status === 'success' && <span className="ml-auto rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-500">{results.length} 条结果</span>}
    </div>
    {status === 'success' ? <div className="space-y-3 p-4">
      <p className="mb-4 px-1 text-xs text-slate-400">按相似程度排序，选择题目查看原文</p>
      {results.map(result => <button type="button" key={result.question_id}
        className={`group w-full rounded-xl border p-3 text-left transition ${selected === result.question_id ? 'border-blue-500 bg-blue-50/40 ring-1 ring-blue-500/20' : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'}`}
        aria-pressed={selected === result.question_id} aria-label={`查看第 ${result.question_no} 题：${result.stem_text || '图形题'}`}
        onClick={() => onSelect(result)}>
        <div className="flex items-start gap-3">
          <div className="w-24 shrink-0 overflow-hidden rounded-lg border border-slate-100 bg-slate-50">
            <RemoteImage url={result.image.preview_url} alt={`第 ${result.question_no} 题缩略图`} onExpired={() => onExpired(result.question_id, result.record_revision)} />
          </div>
          <div className="min-w-0 flex-1">
            <div className="mb-2 flex items-center justify-between gap-2"><span className="text-xs font-medium text-slate-400">结果 {String(result.rank).padStart(2, '0')}</span>
              <span className="review-badge">{staleIds.has(result.question_id) ? '记录已更新' : '待核对'}</span></div>
            <p className="line-clamp-2 break-words text-sm font-medium leading-6 text-slate-800">{result.stem_text || '图形题 · 查看题图'}</p>
            <p className="mt-2 text-xs text-slate-500">第 {result.page_number} 页 <span className="px-1 text-slate-300">/</span> 第 {result.question_no} 题</p>
          </div>
        </div>
        <div className="mt-3 flex items-center gap-2 border-t border-slate-100 pt-2.5 text-xs text-slate-400">
          <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-slate-300" /><span className="truncate">{result.document_id}</span>
          <Icon name="arrow" className={`ml-auto h-3.5 w-3.5 shrink-0 ${selected === result.question_id ? 'text-blue-600' : 'text-slate-300'}`} />
        </div>
      </button>)}
    </div> : <div className="flex min-h-80 flex-col items-center justify-center px-8 py-14 text-center" role="status" aria-live="polite">
      <span className={`mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-50 text-slate-300 ${status === 'loading' ? 'animate-pulse' : ''}`}><Icon name="search" className="h-7 w-7" /></span>
      <p className="text-sm font-medium text-slate-500">{status === 'loading' ? '正在题库中寻找相似题…' : status === 'empty' ? '没有找到有效的相似题' : status === 'error' ? '搜索未完成' : '相似题目会出现在这里'}</p>
      <p className="mt-2 text-xs leading-6 text-slate-400">{status === 'empty' ? '试试更完整的截图，或确认题库已建立索引。' : '选择一张题图，开始这次查找。'}</p>
    </div>}
  </section>
}
