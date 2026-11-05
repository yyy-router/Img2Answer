import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from img2answer.api import ApiSettings, create_app


class WebUiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.dist = self.root / "dist"
        self.model = Mock(side_effect=AssertionError("UI must not load a model"))

    def client(self):
        return TestClient(create_app(settings=ApiSettings(web_dist=self.dist, database=self.root / "missing.db"),
                                     embedding_model_factory=self.model))

    def build(self):
        (self.dist / "assets").mkdir(parents=True)
        (self.dist / "index.html").write_text('<html lang="zh-CN"><h1>以图找题</h1></html>', encoding="utf-8")
        (self.dist / "assets" / "app-hash.js").write_text("console.log('web')", encoding="utf-8")

    def test_missing_build_does_not_break_api_or_create_files(self):
        client = self.client()
        response = client.get("/")
        self.assertEqual(response.status_code, 503)
        self.assertIn("前端尚未构建", response.json()["detail"])
        self.assertNotIn(str(self.root), response.text)
        self.assertEqual(client.get("/docs").status_code, 200)
        self.assertEqual(client.get("/openapi.json").status_code, 200)
        self.assertEqual(client.get("/questions/unknown").status_code, 503)
        self.assertFalse(self.dist.exists())
        self.assertFalse((self.root / "missing.db").exists())
        self.model.assert_not_called()

    def test_serves_built_page_and_assets_with_no_index_cache(self):
        self.build()
        client = self.client()
        index = client.get("/")
        self.assertEqual(index.status_code, 200)
        self.assertIn("text/html", index.headers["content-type"])
        self.assertIn("以图找题", index.text)
        self.assertEqual(index.headers["cache-control"], "no-store")
        self.assertEqual(client.get("/assets/app-hash.js").text, "console.log('web')")
        self.model.assert_not_called()

    def test_unknown_paths_and_assets_do_not_fall_back_to_spa(self):
        self.build()
        client = self.client()
        for path in ("/not-an-api", "/search/unknown", "/assets/missing.js", "/assets/%2e%2e/index.html"):
            with self.subTest(path=path):
                self.assertEqual(client.get(path).status_code, 404)
        self.assertEqual(client.get("/docs").status_code, 200)
        self.assertEqual(client.get("/openapi.json").status_code, 200)

    def test_static_assets_never_expose_the_repository(self):
        self.build()
        (self.root / "private.txt").write_text("private test data", encoding="utf-8")
        self.assertEqual(self.client().get("/assets/%2e%2e/%2e%2e/private.txt").status_code, 404)

    def test_web_dist_file_process_and_explicit_configuration(self):
        (self.root / ".env").write_text("IMG2ANSWER_WEB_DIST=custom/frontend\n", encoding="utf-8")
        with patch("img2answer.api.Path.cwd", return_value=self.root), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ApiSettings.from_env().web_dist, Path("custom/frontend"))
            self.assertNotIn("IMG2ANSWER_WEB_DIST", os.environ)
            with patch.dict(os.environ, {"IMG2ANSWER_WEB_DIST": "another/build"}):
                self.assertEqual(ApiSettings.from_env().web_dist, Path("another/build"))
                self.assertEqual(self.client().app.state.settings.web_dist, self.dist)

    def test_default_and_empty_web_dist_configuration(self):
        with patch("img2answer.api.Path.cwd", return_value=self.root), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(ApiSettings.from_env().web_dist, Path("frontend/dist"))
            with patch.dict(os.environ, {"IMG2ANSWER_WEB_DIST": ""}):
                self.assertEqual(ApiSettings.from_env().web_dist, Path("frontend/dist"))


if __name__ == "__main__":
    unittest.main()
