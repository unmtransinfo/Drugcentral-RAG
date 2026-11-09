"""Bearer authentication and Swagger contracts without models or databases."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api import app


class BearerAuthTests(unittest.TestCase):
    protected_routes = (
        ("post", "/api/chat"),
        ("get", "/api/metrics"),
        ("get", "/api/evaluation/info"),
        ("post", "/api/evaluation/jobs"),
        ("get", "/api/evaluation/jobs/00000000-0000-0000-0000-000000000000"),
    )

    def setUp(self):
        self.client = TestClient(app)  # Skip model-loading lifespan.
        self.env = patch.dict(os.environ, {
            "API_AUTH_TOKEN": "test-secret", "ENABLE_EVALUATION_API": "true",
        })
        self.env.start()

    def tearDown(self):
        self.client.close()
        self.env.stop()

    def request(self, method, path, headers):
        kwargs = {"headers": headers}
        if method == "post":
            kwargs["json"] = {"question": "hi"}
        return self.client.request(method, path, **kwargs)

    def test_missing_invalid_and_malformed_tokens_on_every_protected_route(self):
        for method, path in self.protected_routes:
            for header in (None, "Bearer wrong", "Basic test-secret", "Bearer", "Bearer "):
                with self.subTest(path=path, header=header):
                    response = self.request(method, path, {} if header is None else {
                        "Authorization": header,
                    })
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.json(), {"detail": "Unauthorized"})
                    self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")
                    self.assertNotIn("test-secret", response.text)

    def test_valid_token_preserves_case_insensitive_scheme_and_whitespace(self):
        with patch.object(app.state, "fast_answer", lambda q: "ok", create=True):
            for header in ("Bearer test-secret", "bearer test-secret", "Bearer  test-secret "):
                with self.subTest(header=header):
                    headers = {"Authorization": header}
                    self.assertEqual(self.request("post", "/api/chat", headers).json(), {"answer": "ok"})
                    self.assertEqual(self.client.get("/api/metrics", headers=headers).status_code, 200)
                    self.assertEqual(self.client.get("/api/evaluation/info", headers=headers).status_code, 200)

    def test_unconfigured_token_keeps_chat_and_metrics_optional(self):
        with patch.dict(os.environ, {"API_AUTH_TOKEN": ""}), \
                patch.object(app.state, "fast_answer", lambda q: "ok", create=True):
            for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic wrong"}):
                self.assertEqual(self.request("post", "/api/chat", headers).status_code, 200)
                self.assertEqual(self.client.get("/api/metrics", headers=headers).status_code, 200)

    def test_evaluation_restrictions_precede_token_validation(self):
        for method, path in self.protected_routes[2:]:
            for headers in ({}, {"Authorization": "Bearer wrong"},
                            {"Authorization": "Bearer test-secret"}):
                with self.subTest(path=path, headers=headers):
                    with patch.dict(os.environ, {"ENABLE_EVALUATION_API": "false"}):
                        self.assertEqual(self.request(method, path, headers).status_code, 404)
                    with patch.dict(os.environ, {"API_AUTH_TOKEN": ""}):
                        response = self.request(method, path, headers)
                        self.assertEqual(response.status_code, 503)
                        self.assertEqual(response.json(), {"detail": "Evaluation API requires API_AUTH_TOKEN"})

    def test_health_and_ready_remain_public_with_token_configured(self):
        index = SimpleNamespace(ids={"demo": 1}, versions={"test"})
        with patch.object(app.state, "evidence_index", index, create=True), \
                patch.object(app.state, "chat_backend", "bedrock", create=True), \
                patch("chatbot.storage._connect", MagicMock()):  # readiness reads the database; no real connection
            self.assertEqual(self.client.get("/api/health").status_code, 200)
            ready = self.client.get("/api/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["provider"], "bedrock")

    def test_local_mode_ready_requires_the_ollama_chat_model(self):
        index = SimpleNamespace(ids={"demo": 1}, versions={"test"})
        tags = MagicMock(**{"json.return_value": {"models": [{"name": "qwen3.5:9b"}]}})
        with patch.object(app.state, "evidence_index", index, create=True), \
                patch.object(app.state, "chat_backend", "ollama", create=True), \
                patch("chatbot.storage._connect", MagicMock()), patch("api.httpx.get", return_value=tags):
            for model, status in [("qwen3.5:9b", 200), ("qwen3.5:4b", 503)]:
                with self.subTest(model=model), patch.object(app.state, "chat_model", model, create=True):
                    ready = self.client.get("/api/ready")
                    self.assertEqual(ready.status_code, status)
                    if status == 200:
                        self.assertEqual(ready.json()["provider"], "ollama")
                    else:
                        self.assertEqual(ready.json(), {"detail": "Ollama is not ready"})

    def test_openapi_declares_bearer_on_all_protected_operations(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        schema = response.json()
        self.assertEqual(schema["components"]["securitySchemes"]["HTTPBearer"],
                         {"type": "http", "scheme": "bearer"})
        for method, path in self.protected_routes:
            if path.startswith("/api/evaluation/jobs/"):
                path = "/api/evaluation/jobs/{job_id}"
            self.assertEqual(schema["paths"][path][method]["security"], [{"HTTPBearer": []}])
        for path in ("/api/health", "/api/ready"):
            self.assertNotIn("security", schema["paths"][path]["get"])
        self.assertNotIn("test-secret", response.text)
        docs = self.client.get("/docs")
        self.assertEqual(docs.status_code, 200)
        self.assertIn("SwaggerUIBundle", docs.text)
        self.assertIn("/openapi.json", docs.text)


if __name__ == "__main__":
    unittest.main()
