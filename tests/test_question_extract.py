from contextlib import closing, redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import pymupdf

from img2answer import cli, ocr
from img2answer.graphic_crop import CropCandidate
from img2answer.pdf_document import PdfMetadata
from img2answer.question_extract import ExtractionError, extract_page, extract_questions
from img2answer.report import SectionReport, build_report, write_report
from img2answer.store import SQLiteStore

from helpers import create_sample_pdf
from test_ocr import RecordingOCRClient, settings


def block(text, bbox, confidence=95):
    left, top, right, bottom = bbox
    return dict(text=text, bbox=list(bbox), confidence=confidence,
                polygon=[[left, top], [right, top], [right, bottom], [left, bottom]])


def sample_blocks():
    return [block("HEADER", (20, 5, 95, 20), confidence=1),
            block("1.", (20, 40, 35, 55)), block("第一道题", (45, 36, 180, 58)),
            block("A. first B. second C. third D. fourth", (20, 150, 350, 165)),
            block("答案：", (20, 180, 55, 195)), block("A", (60, 180, 70, 195)),
            block("解析：示例", (20, 205, 150, 220)),
            block("2)", (20, 300, 35, 315)), block("第二道题", (45, 296, 180, 318)),
            block("正确答案：B", (20, 450, 150, 465))]


def make_snapshot(blocks, graphics=(), size=(400, 600)):
    image = Image.new("RGB", size, "white")
    drawing = ImageDraw.Draw(image)
    for item in blocks:
        left, top, right, bottom = item["bbox"]
        drawing.rectangle((left, top, right - 1, bottom - 1), fill="black")
    for box in graphics:
        drawing.rectangle(box, fill="black")
    data = io.BytesIO()
    image.save(data, format="PNG")
    return ocr.OCRSnapshot(data.getvalue(), *size, {"blocks": blocks})


def response_for(snapshot):
    words = [{"word": item["text"], "prob": item["confidence"],
              "pos": [{"x": x, "y": y} for x, y in item["polygon"]]}
             for item in snapshot.result["blocks"]]
    return {"RequestId": "local-sample", "Data": json.dumps({
        "width": snapshot.width, "height": snapshot.height,
        "content": "\n".join(item["text"] for item in snapshot.result["blocks"]),
        "prism_wordsInfo": words,
    }, ensure_ascii=False)}


class QuestionPageTests(unittest.TestCase):
    def page(self, blocks=None, graphics=()):
        return extract_page(make_snapshot(sample_blocks() if blocks is None else blocks, graphics), page_number=1)

    def test_two_questions_use_line_top_and_keep_unrecognized_graphics(self):
        snapshot = make_snapshot(sample_blocks(), [(70, 70, 330, 130), (60, 500, 340, 530)])
        result = extract_page(snapshot, page_number=1)
        first, second = result["candidates"]
        self.assertEqual([first["question_no"], second["question_no"]], ["1", "2"])
        self.assertEqual(first["bbox"][1], 36)
        self.assertLessEqual(first["bbox"][3], 296)
        self.assertEqual(second["bbox"][1], 296)
        self.assertGreater(second["bbox"][3], 530)
        self.assertEqual(first["answer"], "A")
        self.assertEqual(first["options"], dict(A="first", B="second", C="third", D="fourth"))
        self.assertEqual(first["stem_text"], "第一道题")
        self.assertEqual(first["analysis_text"], "示例")
        self.assertEqual(second["answer"], "B")
        self.assertEqual(second["options"], {})
        self.assertIn("open_page_end", second["review_reasons"])
        self.assertNotIn("low_confidence", first["review_reasons"])
        self.assertIn("unassigned_prefix", result["warnings"])

    def test_ocr_array_order_does_not_change_candidates(self):
        self.assertEqual(self.page(sample_blocks()), self.page(list(reversed(sample_blocks()))))

    def test_supported_inline_question_numbers(self):
        for text in ("1.题干", "1．题干", "1、题干", "1)题干", "9999）题干"):
            with self.subTest(text=text):
                result = self.page([block(text, (20, 30, 180, 50))])
                self.assertEqual(len(result["candidates"]), 1)
                self.assertEqual(result["candidates"][0]["stem_text"], "题干")

    def test_decimal_date_plain_numbers_and_options_are_not_anchors(self):
        for text in ("1.2", "1. 2", "2026.11.02", "1", "0.题干", "10000.题干", "A.题干"):
            with self.subTest(text=text):
                result = self.page([block(text, (20, 30, 180, 50))])
                self.assertEqual(result["candidates"], [])
                self.assertIn("no_question_anchor", result["warnings"])

    def test_conflicting_anchors_skip_entire_page(self):
        samples = [([block("1.题干", (20, 30, 180, 50)), block("1.题干", (20, 80, 180, 100))], "duplicate_question_number"),
                   ([block("2.题干", (20, 30, 180, 50)), block("1.题干", (20, 80, 180, 100))], "question_number_order"),
                   ([block("1.题干", (20, 30, 100, 50)), block("2.题干", (200, 30, 380, 50))], "multiple_anchors_on_line"),
                   ([block("1.题干", (20, 30, 180, 50)), block("2.题干", (60, 80, 200, 100))], "unsupported_columns")]
        for blocks, reason in samples:
            with self.subTest(reason=reason):
                result = self.page(blocks)
                self.assertEqual(result["candidates"], [])
                self.assertIn(reason, result["warnings"])

    def test_missing_number_is_flagged_without_inventing_a_question(self):
        result = self.page([block("1.题干", (20, 30, 180, 50)), block("3.题干", (20, 80, 180, 100))])
        self.assertEqual(len(result["candidates"]), 2)
        self.assertIn("question_number_gap", result["candidates"][1]["review_reasons"])

    def test_merged_multiquestion_box_has_no_invented_coordinates(self):
        result = self.page([block("1.题干\n2.题干", (20, 30, 180, 100))])
        self.assertEqual(result["candidates"], [])
        self.assertIn("ambiguous_boundary", result["warnings"])

    def test_crossing_text_block_is_ambiguous(self):
        result = self.page([block("1.题干", (20, 30, 180, 50)),
                            block("跨界", (250, 70, 300, 190)), block("2.题干", (20, 100, 180, 120))])
        self.assertIn("ambiguous_boundary", result["warnings"])
        self.assertEqual(result["candidates"], [])

    def test_number_only_tail_is_not_a_question(self):
        result = self.page([block("1.", (20, 550, 35, 570))])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(result["skipped_candidates"][0]["reason"], "no_question_content")

    def test_graphic_only_question_is_retained_with_empty_text(self):
        result = self.page([block("1.", (20, 30, 35, 50))], [(50, 80, 300, 200)])
        candidate = result["candidates"][0]
        self.assertEqual(candidate["stem_text"], "")
        self.assertEqual(candidate["options"], {})
        self.assertIsNone(candidate["answer"])
        self.assertIn("missing_stem_text", candidate["review_reasons"])

    def test_empty_page_has_an_explicit_reason(self):
        self.assertIn("empty_page", self.page([])["warnings"])

    def test_candidate_confidence_boundaries(self):
        for confidence, expected in ((None, "missing_confidence"), (79, "low_confidence"), (80, None)):
            with self.subTest(confidence=confidence):
                candidate = self.page([block("1.题干", (20, 30, 180, 50), confidence)])["candidates"][0]
                if expected:
                    self.assertIn(expected, candidate["review_reasons"])
                else:
                    self.assertNotIn("low_confidence", candidate["review_reasons"])
                    self.assertNotIn("missing_confidence", candidate["review_reasons"])

    def test_answers_require_adjacent_same_line_value_and_explicit_label(self):
        tails = [([block("答案", (20, 80, 50, 100)), block("C", (55, 80, 65, 100))], "C"),
                 ([block("答案", (20, 80, 50, 100)), block("C", (55, 110, 65, 130))], None),
                 ([block("答案", (20, 80, 50, 100)), block("C", (300, 80, 315, 100))], None),
                 ([block("C", (20, 80, 30, 100))], None),
                 ([block("答案：AB", (20, 80, 130, 100))], None),
                 ([block("答案：A/B", (20, 80, 130, 100))], None)]
        for tail, answer in tails:
            with self.subTest(tail=tail):
                candidate = self.page([block("1.题干", (20, 30, 180, 50)), *tail])["candidates"][0]
                self.assertEqual(candidate["answer"], answer)

    def test_conflicting_answers_and_duplicate_options_are_not_silently_selected(self):
        candidate = self.page([block("1.题干", (20, 30, 180, 50)),
                               block("A. one A. two", (20, 70, 200, 90)),
                               block("答案：A 正确答案：B", (20, 110, 300, 130))])["candidates"][0]
        self.assertIsNone(candidate["answer"])
        self.assertEqual(candidate["options"], {})
        self.assertIn("conflicting_answer", candidate["review_reasons"])
        self.assertIn("conflicting_options", candidate["review_reasons"])

    def test_multiline_option_and_same_line_analysis(self):
        candidate = self.page([block("1.题干", (20, 30, 180, 50)),
                               block("A. 第一行", (20, 70, 180, 90)), block("续行", (20, 100, 180, 120)),
                               block("B. 第二项", (20, 130, 180, 150)),
                               block("答案： A 解析：示例", (20, 170, 300, 190))])["candidates"][0]
        self.assertEqual(candidate["options"]["A"], "第一行\n续行")
        self.assertEqual(candidate["answer"], "A")
        self.assertEqual(candidate["analysis_text"], "示例")

    def test_marker_words_inside_stem_are_not_field_labels(self):
        candidate = self.page([block("1.请选择正确答案并说明解析过程", (20, 30, 300, 50))])["candidates"][0]
        self.assertEqual(candidate["stem_text"], "请选择正确答案并说明解析过程")
        self.assertIsNone(candidate["answer"])
        self.assertEqual(candidate["analysis_text"], "")

    def test_letter_inside_stem_is_not_an_option_label(self):
        candidate = self.page([block("1.根据 A 模式判断图形", (20, 30, 300, 50))])["candidates"][0]
        self.assertEqual(candidate["stem_text"], "根据 A 模式判断图形")
        self.assertEqual(candidate["options"], {})

    def test_single_question_multiline_block_keeps_its_geometry(self):
        candidate = self.page([block("1.题干\n续行\n答案：A", (20, 30, 180, 110))])["candidates"][0]
        self.assertEqual(candidate["stem_text"], "题干\n续行")
        self.assertEqual(candidate["answer"], "A")


class QuestionExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "questions"
        self.report_path = self.root / "reports" / "sample.json"
        self.store = SQLiteStore(self.root / "metadata.sqlite3")
        self.store.initialize()
        self.metadata = PdfMetadata(self.root / "sample.pdf", "a" * 64, 2)
        self.references = []
        self.snapshot = make_snapshot(sample_blocks(), [(70, 70, 330, 130), (60, 500, 340, 530)])
        self.add_page(1, self.snapshot)
        self.add_page(2, make_snapshot([]))
        self.import_report()

    def add_page(self, number, snapshot):
        image = self.root / "pages" / f"page-{number:04d}.png"
        image.parent.mkdir(exist_ok=True)
        image.write_bytes(snapshot.image)
        result_path = self.root / "ocr" / f"page-{number:04d}.json"
        result = ocr.recognize_page(image, result_path, client=RecordingOCRClient(response=response_for(snapshot)))
        reference = dict(section="graphic_reasoning", page_number=number, source_page=str(image),
                         result_path=str(result_path), image_sha256=result.image_sha256,
                         result_sha256=result.result_sha256, status=result.status)
        self.references = [item for item in self.references if item["page_number"] != number] + [reference]

    def import_report(self):
        self.report = build_report(document_id="sample", metadata=self.metadata,
            sections=[SectionReport("graphic_reasoning", 1, 2, 2, 1)], crop_candidates=[], ocr_pages=self.references)
        write_report(self.report, self.report_path)
        crop = self.root / "old-crop.png"
        crop.write_bytes(self.snapshot.image)
        self.store.replace_document_run(document_id="sample", metadata=self.metadata,
            section_candidates={"graphic_reasoning": [CropCandidate(Path(self.references[0]["source_page"]), crop,
                                                                    (0, 0, 400, 600), 400, 600)]},
            report_path=self.report_path, report=self.report)

    def run_extract(self, **kwargs):
        return extract_questions(self.report_path, self.store.path, self.output, **kwargs)

    def state(self):
        return {table: [dict(row) for row in self.store.fetch_all(table)] for table in
                ("source_documents", "process_reports", "question_images", "questions")}

    def protect_existing(self):
        result = self.run_extract()
        state = self.state()
        files = {Path(question["image"]["output_path"]): Path(question["image"]["output_path"]).read_bytes()
                 for question in result["questions"]}
        files[Path(result["extraction_path"])] = Path(result["extraction_path"]).read_bytes()
        return result, state, files

    def assert_preserved(self, state, files):
        self.assertEqual(self.state(), state)
        for path, content in files.items():
            self.assertEqual(path.read_bytes(), content)

    def test_extract_persists_linked_questions_and_pixel_exact_screenshots(self):
        result = self.run_extract()
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["needs_review_count"], 2)
        self.assertEqual(result["skipped_page_count"], 1)
        self.assertEqual(len(self.store.fetch_all("questions")), 2)
        self.assertEqual(self.store.counts().question_images, 3)
        self.assertEqual(json.loads(Path(result["extraction_path"]).read_text(encoding="utf-8")), result)
        for question in result["questions"]:
            record = self.store.fetch_question(question["id"])
            self.assertEqual(record["image_id"], question["image_id"])
            self.assertEqual(self.store.fetch_question_by_image(question["image_id"])["id"], question["id"])
            self.assertEqual(record["parse_status"], "needs_review")
            self.assertEqual(json.loads(record["options_json"]), question["options"])
            with Image.open(io.BytesIO(self.snapshot.image)) as source, Image.open(question["image"]["output_path"]) as crop:
                self.assertEqual(crop.tobytes(), source.crop(question["image"]["bbox"]).tobytes())

    def test_rerun_replaces_paths_and_records_without_duplicates(self):
        first, state, files = self.protect_existing()
        second = self.run_extract()
        self.assertEqual([q["id"] for q in first["questions"]], [q["id"] for q in second["questions"]])
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(len(self.store.fetch_all("questions")), 2)
        self.assertEqual(self.store.counts().question_images, 3)
        self.assertEqual(state["source_documents"], self.state()["source_documents"])
        self.assertEqual(state["process_reports"], self.state()["process_reports"])
        for path, content in files.items():
            self.assertEqual(path.read_bytes(), content)

    def test_later_page_preflight_failure_preserves_everything(self):
        first, state, files = self.protect_existing()
        Path(self.references[1]["result_path"]).write_text("{", encoding="utf-8")
        with self.assertRaises(ExtractionError):
            self.run_extract()
        self.assert_preserved(state, files)
        self.assertEqual(list(self.output.iterdir()), [Path(first["run_dir"])])

    def test_snapshot_and_cache_changed_together_are_rejected_by_old_report(self):
        self.add_page(1, make_snapshot([block("1.其他题", (20, 30, 180, 50))]))
        with self.assertRaisesRegex(ExtractionError, "快照"):
            self.run_extract()
        self.assertFalse(self.output.exists())

    def test_cache_changed_with_same_image_is_rejected_by_result_digest(self):
        path = Path(self.references[0]["result_path"])
        value = json.loads(path.read_text(encoding="utf-8"))
        value["raw_data"]["prism_wordsInfo"][0]["prob"] = 99
        value["blocks"][0]["confidence"] = 99
        path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ExtractionError, "快照"):
            self.run_extract()

    def test_tampered_normalized_cache_and_raw_geometry_are_rejected(self):
        path = Path(self.references[0]["result_path"])
        original = path.read_bytes()
        for mutate in (lambda v: v["blocks"][0].update(text="changed"),
                       lambda v: v["raw_data"]["prism_wordsInfo"][0]["pos"][0].update(x=-1)):
            value = json.loads(original)
            mutate(value)
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ExtractionError):
                self.run_extract()
        self.assertFalse(self.output.exists())

    def test_old_reports_without_snapshot_binding_are_explicitly_rejected(self):
        for remove in (lambda v: v.pop("imported_at"), lambda v: v["ocr_pages"][0].pop("result_sha256")):
            report = deepcopy(self.report)
            remove(report)
            write_report(report, self.report_path)
            with self.assertRaisesRegex(ExtractionError, "旧报告"):
                self.run_extract()
        self.assertFalse(self.output.exists())

    def test_invalid_report_pages_and_wrong_dataset_are_rejected_without_writes(self):
        changes = [lambda r: r["ocr_pages"].append(deepcopy(r["ocr_pages"][0])),
                   lambda r: r["ocr_pages"].pop(), lambda r: r.update(sha256="b" * 64),
                   lambda r: r["ocr_pages"][0].update(page_number=True),
                   lambda r: r["processed_sections"][0].update(page_to=3),
                   lambda r: r["ocr_pages"][0].update(source_page="")]
        for change in changes:
            with self.subTest(change=change):
                report = deepcopy(self.report)
                change(report)
                write_report(report, self.report_path)
                with self.assertRaises(ExtractionError):
                    self.run_extract()
        self.assertFalse(self.output.exists())

    def test_missing_database_is_not_created(self):
        missing = self.root / "missing.sqlite3"
        with self.assertRaises(ExtractionError):
            extract_questions(self.report_path, missing, self.output)
        self.assertFalse(missing.exists())

    def test_threshold_errors_have_no_side_effects(self):
        for value in (-1, 101, float("nan"), float("inf"), True, 10 ** 400):
            with self.subTest(value=value), self.assertRaises(ExtractionError):
                self.run_extract(min_confidence=value)
        self.assertFalse(self.output.exists())

    def test_output_cannot_be_nested_in_original_artifact_directories(self):
        for name in ("pages", "crops", "ocr", "reports"):
            with self.subTest(name=name), self.assertRaises(ExtractionError):
                extract_questions(self.report_path, self.store.path, self.root / name / "new")
        self.assertFalse(self.output.exists())

    def test_image_write_failure_keeps_history_and_cleans_partial_outputs(self):
        first, state, files = self.protect_existing()
        real_save = Image.Image.save
        counter = [0]
        def fail_second(image, fp, *args, **kwargs):
            counter[0] += 1
            if counter[0] == 2:
                Path(fp).write_bytes(b"partial")
                raise OSError("private diagnostics")
            return real_save(image, fp, *args, **kwargs)
        with patch.object(Image.Image, "save", fail_second), self.assertRaises(ExtractionError) as raised:
            self.run_extract()
        self.assertNotIn("private diagnostics", str(raised.exception))
        self.assert_preserved(state, files)
        self.assertEqual(list(self.output.iterdir()), [Path(first["run_dir"])])

    def test_sqlite_failure_rolls_back_deletes_and_inserts(self):
        first, state, files = self.protect_existing()
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("CREATE TRIGGER fail_questions BEFORE INSERT ON questions BEGIN SELECT RAISE(ABORT, 'fail'); END")
        with self.assertRaises(ExtractionError):
            self.run_extract()
        self.assert_preserved(state, files)
        self.assertEqual(list(self.output.iterdir()), [Path(first["run_dir"])])

    def test_json_write_failure_keeps_history_and_removes_partial_outputs(self):
        first, state, files = self.protect_existing()
        real_write = Path.write_text
        def fail_export(path, text, *args, **kwargs):
            if path.name == "extraction.json":
                path.write_bytes(b"partial")
                raise OSError("local failure")
            return real_write(path, text, *args, **kwargs)
        with patch.object(Path, "write_text", fail_export), self.assertRaises(ExtractionError):
            self.run_extract()
        self.assert_preserved(state, files)
        self.assertEqual(list(self.output.iterdir()), [Path(first["run_dir"])])

    def test_changing_input_path_after_preflight_does_not_change_cropped_bytes(self):
        real_extract = extract_page
        def change_source(snapshot, **kwargs):
            Path(self.references[0]["source_page"]).write_bytes(make_snapshot([]).image)
            return real_extract(snapshot, **kwargs)
        with patch("img2answer.question_extract.extract_page", side_effect=change_source):
            result = self.run_extract()
        for question in result["questions"]:
            with Image.open(io.BytesIO(self.snapshot.image)) as original, Image.open(question["image"]["output_path"]) as crop:
                self.assertEqual(crop.tobytes(), original.crop(question["image"]["bbox"]).tobytes())

    def test_report_change_during_parsing_prevents_writes(self):
        first, state, files = self.protect_existing()
        real_extract = extract_page
        def change_report(snapshot, **kwargs):
            write_report({**self.report, "changed": True}, self.report_path)
            return real_extract(snapshot, **kwargs)
        with patch("img2answer.question_extract.extract_page", side_effect=change_report), self.assertRaises(ExtractionError):
            self.run_extract()
        self.assert_preserved(state, files)
        self.assertEqual(list(self.output.iterdir()), [Path(first["run_dir"])])

    def test_threshold_only_changes_review_flags_and_preserves_inputs(self):
        paths = [self.report_path, *[Path(ref[key]) for ref in self.references for key in ("source_page", "result_path")]]
        before = {path: path.read_bytes() for path in paths}
        result = self.run_extract(min_confidence=99)
        self.assertTrue(all("low_confidence" in question["review_reasons"] for question in result["questions"]))
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    def test_cleanup_failure_is_reported_and_does_not_remove_history(self):
        _, state, files = self.protect_existing()
        with patch.object(SQLiteStore, "replace_extracted_questions", side_effect=sqlite3.OperationalError("fail")), \
                patch.object(Path, "unlink", side_effect=PermissionError("private diagnostics")), \
                self.assertRaisesRegex(ExtractionError, "未完全清理"):
            self.run_extract()
        self.assert_preserved(state, files)

    def test_reimport_cascades_questions_without_deleting_other_documents(self):
        result = self.run_extract()
        other = deepcopy(result["questions"][0])
        other.update(id="other-question", document_id="other", image_id="other-image")
        other_report_path = self.root / "reports" / "other.json"
        other_report = build_report(document_id="other", metadata=self.metadata,
            sections=[SectionReport("graphic_reasoning", 1, 1, 1, 0)], crop_candidates=[])
        write_report(other_report, other_report_path)
        self.store.replace_document_run(document_id="other", metadata=self.metadata, section_candidates={},
            report_path=other_report_path, report=other_report)
        self.store.replace_extracted_questions(self.store.read_import_state("other"), [other])
        other_before = dict(self.store.fetch_question("other-question"))
        self.run_extract()
        self.assertEqual(dict(self.store.fetch_question("other-question")), other_before)
        self.import_report()
        self.assertEqual([row["id"] for row in self.store.fetch_all("questions")], ["other-question"])
        self.assertEqual(dict(self.store.fetch_question("other-question")), other_before)
        self.assertEqual(self.store.counts().question_images, 2)
        self.assertTrue(Path(result["extraction_path"]).exists())

    def test_import_change_after_preflight_is_rejected_in_final_transaction(self):
        first = self.run_extract()
        real_initialize = self.store.initialize
        def initialize_and_change():
            real_initialize()
            self.import_report()
        with patch.object(SQLiteStore, "initialize", side_effect=initialize_and_change), self.assertRaises(ExtractionError):
            self.run_extract()
        self.assertEqual(self.store.fetch_all("questions"), [])
        self.assertEqual(self.store.counts().question_images, 1)
        self.assertEqual(self.store.read_import_state("sample").report_created_at, self.report["imported_at"])
        self.assertEqual(list(self.output.iterdir()), [Path(first["run_dir"])])

    def test_offline_cli_does_not_load_credentials_cloud_or_vector_backends(self):
        out = io.StringIO()
        with patch.object(cli.OCRSettings, "from_env", side_effect=AssertionError("credentials")), \
                patch.object(cli, "AliyunOCRClient", side_effect=AssertionError("cloud")), \
                patch.object(cli, "ChromaImageVectorStore", side_effect=AssertionError("vectors")), \
                patch.object(cli, "create_embedding_model", side_effect=AssertionError("model")), redirect_stdout(out):
            status = cli.main(["extract-questions", "--report", str(self.report_path),
                               "--database", str(self.store.path), "--output-dir", str(self.output)])
        self.assertEqual(status, 0)
        self.assertIn("候选 2 道", out.getvalue())
        self.assertIn("核对文件", out.getvalue())

    def test_cli_error_returns_nonzero_and_safe_chinese_error(self):
        err = io.StringIO()
        with redirect_stderr(err):
            status = cli.main(["extract-questions", "--report", str(self.report_path),
                               "--database", str(self.store.path), "--output-dir", str(self.output),
                               "--min-confidence", "nan"])
        self.assertEqual(status, 1)
        self.assertIn("核对阈值", err.getvalue())

    def test_legacy_database_gets_new_table_only_after_valid_preflight(self):
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("DROP TABLE questions")
        self.run_extract()
        self.assertEqual(len(self.store.fetch_all("questions")), 2)

    def test_failed_preflight_does_not_upgrade_legacy_database(self):
        with closing(sqlite3.connect(self.store.path)) as connection, connection:
            connection.execute("DROP TABLE questions")
        self.report.pop("imported_at")
        write_report(self.report, self.report_path)
        with self.assertRaises(ExtractionError):
            self.run_extract()
        with closing(sqlite3.connect(self.store.path)) as connection:
            self.assertIsNone(connection.execute("SELECT name FROM sqlite_master WHERE name = 'questions'").fetchone())
        self.assertFalse(self.output.exists())

    def test_later_target_range_preserves_original_page_number(self):
        self.add_page(2, self.snapshot)
        self.import_report()
        self.report["processed_sections"] = [SectionReport("graphic_reasoning", 2, 2, 1, 1).__dict__]
        self.report["ocr_pages"] = [ref for ref in self.references if ref["page_number"] == 2]
        write_report(self.report, self.report_path)
        result = self.run_extract()
        self.assertEqual(len(result["questions"]), 2)
        self.assertTrue(all(question["page_number"] == 2 for question in result["questions"]))
        self.assertTrue(all("page-0002" in question["id"] for question in result["questions"]))

    def test_valid_zero_candidate_result_replaces_old_questions_only(self):
        self.run_extract()
        # Keep the same imported source while replacing its page and bound OCR snapshot.
        self.add_page(1, make_snapshot([]))
        self.report["ocr_pages"] = self.references
        write_report(self.report, self.report_path)
        state = self.state()
        result = self.run_extract()
        self.assertEqual(result["candidate_count"], 0)
        self.assertEqual(self.store.fetch_all("questions"), [])
        self.assertEqual(self.store.counts().question_images, 1)
        self.assertEqual(state["source_documents"], self.state()["source_documents"])
        self.assertEqual(state["process_reports"], self.state()["process_reports"])


class ImportSnapshotRegressionTests(unittest.TestCase):
    def test_failure_in_other_section_cannot_relabel_changed_target_as_old_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, changed = root / "original.pdf", root / "changed.pdf"
            create_sample_pdf(original)
            config, output = root / "sections.json", root / "processed"
            def configure(pdf):
                config.write_text(json.dumps({"documents": {"sample": {"path": str(pdf), "sections": {
                    "graphic_reasoning": {"page_from": 1, "page_to": 1},
                    "other": {"page_from": 2, "page_to": 2}}}}}), encoding="utf-8")
            configure(original)
            args = ["--config", str(config), "--output-dir", str(output), "--dpi", "96", "--ocr"]
            with patch.object(cli.OCRSettings, "from_env", return_value=settings()), \
                    patch.object(cli, "AliyunOCRClient", return_value=RecordingOCRClient()), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(args), 0)
            report_path = output / "reports" / "sample-report.json"
            old_report = report_path.read_bytes()
            with pymupdf.open(original) as pdf:
                pdf[0].draw_circle(pymupdf.Point(120, 80), 12)
                pdf[1].draw_rect(pymupdf.Rect(50, 50, 100, 100))
                pdf.save(changed)
            configure(changed)
            client = RecordingOCRClient()
            recognize = client.recognize
            count = [0]
            def fail_second(stream):
                count[0] += 1
                if count[0] == 2:
                    raise ocr.OCRError("local simulated failure")
                return recognize(stream)
            with patch.object(cli.OCRSettings, "from_env", return_value=settings()), \
                    patch.object(cli, "AliyunOCRClient", return_value=client), \
                    patch.object(client, "recognize", side_effect=fail_second), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args), 1)
            self.assertEqual(report_path.read_bytes(), old_report)
            report = json.loads(old_report)
            ref = report["ocr_pages"][0]
            snapshot = ocr.read_ocr_snapshot(ref["source_page"], ref["result_path"])
            self.assertNotEqual(snapshot.result["image_sha256"], ref["image_sha256"])
            with self.assertRaisesRegex(ExtractionError, "快照"):
                extract_questions(report_path, output / "img2answer.sqlite3", output / "questions")
            self.assertFalse((output / "questions").exists())


if __name__ == "__main__":
    unittest.main()
