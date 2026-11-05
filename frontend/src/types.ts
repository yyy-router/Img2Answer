export interface Question {
  question_id: string
  image_id: string
  record_revision: string
  document_id: string
  section: string
  page_number: number
  question_no: string
  stem_text: string
  options: Record<string, string>
  answer: 'A' | 'B' | 'C' | 'D' | null
  analysis_text: string
  parse_status: 'needs_review'
  review_reasons: string[]
  image: { width: number; height: number; bbox: number[]; preview_url: string }
  raw_text?: string
}

export interface SearchResult extends Question { rank: number; distance: number | null }
export type SearchStatus = 'idle' | 'loading' | 'success' | 'empty' | 'error'
export type DetailState =
  | { status: 'idle' | 'loading' }
  | { status: 'error' | 'expired'; message: string }
  | { status: 'ready'; question: Question; updated: boolean }
