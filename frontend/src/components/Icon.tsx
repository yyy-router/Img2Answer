export function Icon({ name, className = 'h-5 w-5' }: {
  name: 'search' | 'image' | 'upload' | 'arrow' | 'check' | 'close' | 'expand'
  className?: string
}) {
  const paths = {
    search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4.5 4.5" /></>,
    image: <><rect x="3" y="3" width="18" height="18" rx="4" /><circle cx="8" cy="8" r="1.2" /><path d="m4 17 5-5 4 4 3-3 5 5" /></>,
    upload: <><path d="M12 16V4m-4 4 4-4 4 4M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" /></>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    expand: <><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5" /></>,
  }
  return <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}
