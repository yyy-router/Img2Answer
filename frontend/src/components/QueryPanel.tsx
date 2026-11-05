import { useRef } from 'react'
import type { ClipboardEvent, FormEvent } from 'react'
import { useFileUrl } from '../hooks'
import { Icon } from './Icon'

export function validTopK(value: string) {
  return !value.trim() || /^\d+$/.test(value.trim()) && Number.isSafeInteger(Number(value)) && Number(value) > 0
}

export function QueryPanel({ file, topK, ready, loading, error, onFile, onTopK, onReady, onSubmit }: {
  file: File | null; topK: string; ready: boolean; loading: boolean; error: string
  onFile: (file: File | null) => void; onTopK: (value: string) => void
  onReady: (file: File, valid: boolean) => void; onSubmit: () => void
}) {
  const input = useRef<HTMLInputElement>(null)
  const url = useFileUrl(file)
  function paste(event: ClipboardEvent) {
    const image = Array.from(event.clipboardData.files).find(item => item.type.startsWith('image/')) ??
      Array.from(event.clipboardData.items).find(item => item.type.startsWith('image/'))?.getAsFile()
    if (image) { event.preventDefault(); onFile(image) }
  }
  function submit(event: FormEvent) { event.preventDefault(); onSubmit() }
  const invalidCount = !validTopK(topK)
  return <section className="panel h-fit" aria-labelledby="query-heading">
    <div className="flex items-center gap-3 border-b border-slate-100 px-5 py-4">
      <span className="step-label">01</span><h2 id="query-heading" className="font-semibold">上传题图</h2>
    </div>
    <form className="space-y-5 p-5" onSubmit={submit} noValidate>
      <div tabIndex={0} role="group" aria-label="截图粘贴区域" onPaste={paste}
        className="rounded-xl border border-dashed border-slate-300 bg-slate-50/60 px-4 py-7 text-center outline-offset-4 focus-visible:outline-2 focus-visible:outline-blue-600">
        {file && url ? <img key={url} src={url} alt="查询图片预览" className="mx-auto max-h-56 max-w-full rounded-md object-contain"
          onLoad={() => onReady(file, true)} onError={() => onReady(file, false)} /> :
          <><span className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-slate-200 bg-white text-slate-500 shadow-sm"><Icon name="upload" /></span>
            <p className="text-sm font-medium text-slate-700">从一张题目截图开始</p>
            <p className="mt-1 text-xs leading-5 text-slate-400">完整题干与选项，让查找更准确</p></>}
        <label className="sr-only" htmlFor="query-file">题目图片</label>
        <input id="query-file" ref={input} type="file" accept="image/*" className="hidden" onChange={event => {
          const chosen = event.currentTarget.files?.[0]
          if (chosen) onFile(chosen)
          event.currentTarget.value = ''
        }} />
        <button type="button" className="button-secondary mx-auto mt-5" onClick={() => input.current?.click()}>
          <Icon name="image" className="h-4 w-4" />{file ? '更换图片' : '选择图片'}
        </button>
        <p className="mt-3 text-xs text-slate-400">也可以聚焦此区域，按 Ctrl+V 粘贴截图</p>
      </div>
      {file && <div className="flex min-w-0 items-center gap-2 rounded-lg bg-slate-50 px-3 py-2 text-xs">
        <Icon name="image" className="h-4 w-4 shrink-0 text-slate-400" />
        <div className="min-w-0 flex-1"><p className="truncate font-medium text-slate-600" title={file.name}>{file.name}</p>
          <p className="mt-0.5 text-slate-400">{file.size < 1024 * 1024 ? `${Math.max(1, Math.round(file.size / 1024))} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</p></div>
        <button type="button" className="rounded p-1.5 text-slate-400 hover:bg-slate-200 hover:text-slate-700" aria-label="清空图片" onClick={() => onFile(null)}><Icon name="close" className="h-4 w-4" /></button>
      </div>}
      <div>
        <label htmlFor="top-k" className="mb-2 block text-sm font-medium text-slate-600">结果数量 <span className="ml-1 font-normal text-slate-400">选填</span></label>
        <input id="top-k" type="number" min="1" step="1" value={topK} disabled={loading}
          placeholder="使用服务默认值" aria-invalid={invalidCount} aria-describedby="top-k-hint"
          onChange={event => onTopK(event.target.value)}
          className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm outline-none transition focus:border-blue-500 focus:ring-3 focus:ring-blue-100 disabled:bg-slate-50" />
        <p id="top-k-hint" className={`mt-1.5 text-xs ${invalidCount ? 'text-red-600' : 'text-slate-400'}`}>{invalidCount ? '请输入大于 0 的整数。' : '留空时使用题库服务的默认数量'}</p>
      </div>
      {error && <p role="alert" className="rounded-lg border border-red-100 bg-red-50 px-3 py-2.5 text-sm leading-6 text-red-700">{error}</p>}
      <button type="submit" className="button-primary w-full justify-center py-3" disabled={!file || !ready || loading || invalidCount}>
        <Icon name="search" className="h-4 w-4" />{loading ? '正在搜索…' : '搜索相似题'}
        {!loading && <Icon name="arrow" className="ml-auto h-4 w-4 opacity-70" />}
      </button>
      <p className="text-xs leading-5 text-slate-400">查询图片仅用于本次搜索。材料中的答案均需结合原题核对。</p>
    </form>
  </section>
}
