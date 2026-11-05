"""Serve generated local test data for browser integration; never calls cloud OCR."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img2answer.api import ApiSettings, create_app
from img2answer.question_index import index_questions
from question_helpers import QuestionFixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = Path(args.data_dir).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("--data-dir 必须是新的或空的测试目录")
    root.mkdir(parents=True, exist_ok=True)
    fixture = QuestionFixture(root)
    index_questions(fixture.store.path, "sample", root / "chroma")
    (root / "query.png").write_bytes(Path(fixture.questions[0]["image"]["output_path"]).read_bytes())
    app = create_app(settings=ApiSettings(database=fixture.store.path, chroma_dir=root / "chroma",
                                         image_root=root, embedding_model="pillow-hash"))
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
