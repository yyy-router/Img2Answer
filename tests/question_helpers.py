from pathlib import Path

from img2answer.graphic_crop import CropCandidate
from img2answer.ocr import recognize_page
from img2answer.pdf_document import PdfMetadata
from img2answer.question_extract import extract_questions
from img2answer.report import SectionReport, build_report, write_report
from img2answer.store import SQLiteStore

from test_ocr import RecordingOCRClient
from test_question_extract import make_snapshot, response_for, sample_blocks


class QuestionFixture:
    """Generated OCR, crops and SQLite records, with no cloud or model downloads."""

    def __init__(self, root: Path, document_id="sample"):
        self.root, self.document_id = root, document_id
        self.store = SQLiteStore(root / "metadata.sqlite3")
        self.store.initialize()
        self.report_path = root / "reports" / f"{document_id}.json"
        self.output = root / "questions" / document_id
        self.snapshot = make_snapshot(
            [item for item in sample_blocks() if not item["text"].startswith("正确答案")],
            [(70, 70, 330, 130), (60, 500, 340, 530)],
        )
        self.import_run()
        self.extract()

    def import_run(self):
        image = self.root / "pages" / self.document_id / "page-0001.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(self.snapshot.image)
        cache = self.root / "ocr" / self.document_id / "page-0001.json"
        result = recognize_page(image, cache, client=RecordingOCRClient(response=response_for(self.snapshot)))
        report = build_report(
            document_id=self.document_id, metadata=PdfMetadata(self.root / "sample.pdf", "a" * 64, 1),
            sections=[SectionReport("graphic_reasoning", 1, 1, 1, 1)], crop_candidates=[],
            ocr_pages=[dict(section="graphic_reasoning", page_number=1, source_page=str(image),
                            result_path=str(cache), status=result.status, image_sha256=result.image_sha256,
                            result_sha256=result.result_sha256)],
        )
        write_report(report, self.report_path)
        self.store.replace_document_run(
            document_id=self.document_id, metadata=PdfMetadata(self.root / "sample.pdf", "a" * 64, 1),
            section_candidates={"graphic_reasoning": [CropCandidate(image, image, (0, 0, 400, 600), 400, 600)]},
            report_path=self.report_path, report=report,
        )

    def extract(self):
        result = extract_questions(self.report_path, self.store.path, self.output)
        self.questions = result["questions"]
        return result

    def update(self, sql, parameters=()):
        with self.store._connect() as connection:
            connection.execute(sql, parameters)

    def clear_questions(self):
        self.update("DELETE FROM question_images WHERE document_id = ? AND image_role = 'question_crop'",
                    (self.document_id,))
