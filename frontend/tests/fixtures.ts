import type { Question, SearchResult } from '../src/types'

export function question(number = 1, overrides: Partial<Question> = {}): Question {
  return {
    question_id: `sample:graphic_reasoning:page-0001:question-00${number}`,
    image_id: `image-${number}`, record_revision: `revision-${number}`, document_id: '图形推理示例',
    section: 'graphic_reasoning', page_number: 1, question_no: String(number),
    stem_text: `示例第 ${number} 题：选择符合规律的图形。`, options: { A: '选项一', B: '选项二' },
    answer: 'A', analysis_text: `第 ${number} 题的解析内容。`, parse_status: 'needs_review',
    review_reasons: ['auto_extracted'], image: { width: 320, height: 180, bbox: [0, 0, 320, 180],
      preview_url: `/image-content/image-${number}?revision=revision-${number}` },
    raw_text: '1. 选择符合规律的图形。\n答案：A', ...overrides,
  }
}

export function hit(number = 1, overrides: Partial<Question> = {}): SearchResult {
  return { ...question(number, overrides), rank: number, distance: number / 10 }
}

export const pngBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII='
