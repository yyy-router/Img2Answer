from pathlib import Path
import tempfile
import unittest

from img2answer.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def test_loads_yaml_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "sections.yml"
            config_path.write_text(
                """
documents:
  sample:
    path: sample.pdf
    sections:
      graphic_reasoning:
        page_from: 1
        page_to: 2
""".strip(),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(len(config.documents), 1)
            document = config.documents[0]
            self.assertEqual(document.document_id, "sample")
            self.assertEqual(document.path, Path("sample.pdf").resolve())
            self.assertEqual(document.sections[0].zero_based_range(), range(0, 2))

    def test_rejects_invalid_page_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "sections.yml"
            config_path.write_text(
                """
documents:
  sample:
    path: sample.pdf
    sections:
      graphic_reasoning:
        page_from: 3
        page_to: 2
""".strip(),
                encoding="utf-8",
            )

            with self.assertRaises(ConfigError):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
