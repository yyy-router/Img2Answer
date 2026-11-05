import { useState } from 'react'
import { useRemoteImage } from '../hooks'
import { Icon } from './Icon'

export function RemoteImage({ url, alt, full = false, onExpired }: {
  url: string; alt: string; full?: boolean; onExpired?: () => void
}) {
  const image = useRemoteImage(url, onExpired)
  const [brokenUrl, setBrokenUrl] = useState<string>()
  const broken = image.status === 'error' || image.src !== undefined && brokenUrl === image.src
  return <div className={full ? 'space-y-2' : 'h-full w-full'}>
    <div className={`image-surface flex items-center justify-center overflow-hidden ${full ? 'min-h-48 rounded-xl border border-slate-200 p-4' : 'h-full min-h-20 rounded-lg'}`}>
      {image.status === 'ready' && !broken ? <img src={image.src} alt={alt}
        className={full ? 'max-h-96 max-w-full object-contain' : 'h-24 w-full object-contain p-1'}
        onError={() => setBrokenUrl(image.src)} /> :
        <div className="flex flex-col items-center gap-2 p-3 text-center text-xs text-slate-400" role="status">
          <Icon name="image" className="h-6 w-6" />
          <span>{broken ? image.message ?? '题图加载失败' : '正在加载题图…'}</span>
        </div>}
    </div>
    {full && image.src && !broken && <a href={image.src} target="_blank" rel="noreferrer"
      className="inline-flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-blue-700">
      <Icon name="expand" className="h-3.5 w-3.5" />查看原图<span className="sr-only">（在新标签页打开）</span>
    </a>}
  </div>
}
