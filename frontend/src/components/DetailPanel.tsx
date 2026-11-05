import type { DetailState } from '../types'
import { Icon } from './Icon'
import { RemoteImage } from './RemoteImage'

const reasons: Record<string, string> = {
  auto_extracted: '由材料自动提取', missing_stem_text: '题干文字缺失', missing_options: '未提取到文字选项',
  missing_answer: '未识别到答案', conflicting_answer: '答案存在冲突', invalid_answer: '答案格式需核对',
  low_confidence: '文字识别置信度较低', missing_confidence: '缺少识别置信度', open_page_end: '题目位于页尾，请核对完整性',
  question_number_gap: '题号不连续', conflicting_options: '选项标签存在冲突',
}

export function DetailPanel({ state, onRefresh, onExpired }: {
  state: DetailState; onRefresh: () => void; onExpired: (revision: string) => void
}) {
  const question = state.status === 'ready' ? state.question : null
  return <section className="panel min-w-0" aria-labelledby="detail-heading" aria-busy={state.status === 'loading'}>
    <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-4">
      <span className="step-label">03</span><h2 id="detail-heading" className="font-semibold">题目详情</h2>
      {question && <span className="review-badge ml-auto">待核对</span>}
    </div>
    {question ? <div className="space-y-6 p-5">
      {state.status === 'ready' && state.updated && <p role="status" className="notice">题目记录已更新。以下是当前详情，请重新搜索以确认匹配。</p>}
      <div><p className="break-words text-sm font-semibold text-slate-700">{question.document_id}</p>
        <p className="mt-1 text-xs text-slate-400">第 {question.page_number} 页 <span className="px-1">·</span> 第 {question.question_no} 题 <span className="px-1">·</span> 来源截图</p></div>
      <RemoteImage key={question.record_revision} url={question.image.preview_url} alt="完整题目截图" full onExpired={() => onExpired(question.record_revision)} />
      <div className="space-y-5">
        <div><h3 className="field-heading">题干</h3><p className="question-text">{question.stem_text || '未提取到题干文字，请以题图为准。'}</p></div>
        <div><h3 className="field-heading">选项</h3>
          {Object.keys(question.options).length ? <div className="space-y-2">{Object.entries(question.options).sort().map(([label, text]) =>
            <div key={label} className="flex items-start gap-3 rounded-lg bg-slate-50 px-3 py-2.5 text-sm leading-6">
              <span className="font-semibold text-slate-400">{label}</span><span className="min-w-0 whitespace-pre-wrap break-words text-slate-600">{text}</span>
            </div>)}</div> : <p className="text-sm leading-6 text-slate-400">图形选项请以题图为准。</p>}
        </div>
        <div className="rounded-xl border border-amber-200/80 bg-amber-50/60 p-4">
          <div className="flex items-center justify-between gap-2"><h3 className="text-sm font-semibold text-slate-700">材料中的答案</h3><span className="review-badge">待核对</span></div>
          <p className={`mt-3 ${question.answer ? 'text-3xl font-semibold text-slate-800' : 'text-sm font-medium text-amber-800'}`}>
            {question.answer ?? (question.review_reasons.includes('conflicting_answer') ? '答案存在冲突，需核对' : '未识别到答案')}</p>
          <p className="mt-2 text-xs leading-5 text-slate-500">答案为自动提取值，请结合来源题图核对。</p>
        </div>
        <div><h3 className="field-heading">解析</h3><p className="question-text">{question.analysis_text || '材料中未提取到解析。'}</p></div>
      </div>
      <div className="border-t border-slate-100 pt-4"><h3 className="field-heading">核对提示</h3><ul className="space-y-1.5 text-xs leading-5 text-slate-500">
        {[...new Set(question.review_reasons.map(reason => reasons[reason] ?? '存在其他待核对项'))].map(reason => <li key={reason} className="flex gap-2"><span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-amber-400" />{reason}</li>)}
      </ul></div>
      {question.raw_text && <details className="border-t border-slate-100 pt-4"><summary className="cursor-pointer text-xs font-medium text-slate-500">查看识别原文</summary><p className="mt-3 whitespace-pre-wrap break-words text-xs leading-6 text-slate-500">{question.raw_text}</p></details>}
    </div> : <div className="flex min-h-96 flex-col items-center justify-center px-7 py-14 text-center" aria-live="polite">
      <span className="mb-5 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-50 text-slate-300"><Icon name="image" className="h-7 w-7" /></span>
      {state.status === 'expired' || state.status === 'error' ? <><p role="alert" className="text-sm leading-7 text-slate-600">{state.message}</p><button className="button-secondary mt-4" onClick={onRefresh}>刷新详情</button></> :
        <><p role="status" className="text-sm font-medium text-slate-500">{state.status === 'loading' ? '正在读取题目详情…' : '选择一道题，核对原文'}</p><p className="mt-2 text-xs leading-6 text-slate-400">完整题图、答案与解析将在这里展示。</p></>}
    </div>}
  </section>
}
