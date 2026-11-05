import { useEffect, useRef, useState } from 'react'
import { ApiError, checkedFetch, errorMessage } from './api'

export function useFileUrl(file: File | null): string | undefined {
  const [value, setValue] = useState<{ file: File; url: string }>()
  useEffect(() => {
    if (!file) { setValue(undefined); return }
    const url = URL.createObjectURL(file)
    setValue({ file, url })
    return () => URL.revokeObjectURL(url)
  }, [file])
  return value?.file === file ? value?.url : undefined
}

type ImageState = { url: string; status: 'loading' | 'ready' | 'error'; src?: string; message?: string }
export function useRemoteImage(url: string, onExpired?: () => void) {
  const [state, setState] = useState<ImageState>({ url, status: 'loading' })
  const expired = useRef(onExpired)
  const currentUrl = useRef(url)
  expired.current = onExpired
  currentUrl.current = url
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    let objectUrl: string | undefined
    setState({ url, status: 'loading' })
    checkedFetch(url, { signal: controller.signal })
      .then(response => response.blob())
      .then(blob => {
        if (!active || currentUrl.current !== url) return
        if (!blob.size || !blob.type.startsWith('image/')) throw new Error('Invalid image response')
        objectUrl = URL.createObjectURL(blob)
        setState({ url, status: 'ready', src: objectUrl })
      })
      .catch(error => {
        if (!active || currentUrl.current !== url) return
        setState({ url, status: 'error', message: errorMessage(error, 'image') })
        if (error instanceof ApiError && error.status === 409) expired.current?.()
      })
    return () => {
      active = false
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [url])
  return state.url === url ? state : { url, status: 'loading' as const }
}
