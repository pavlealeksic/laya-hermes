"""Unit tests for the Laya plugin handlers — backend stubbed, no model download."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_laya import backend, tools  # noqa: E402


def _fake_result():
    return {
        "answers": {
            "answer": {
                "choice": "billing",
                "confidence": 0.91,
                "probabilities": {"billing": 0.91, "technical": 0.06, "other": 0.03},
            }
        },
        "action": {"act_probability": 0.87},
    }


class DecideTests(unittest.TestCase):
    def test_requires_state(self):
        out = json.loads(tools.handle_decide({"questions": {"q": {"type": "noul"}}}))
        self.assertFalse(out["success"])
        self.assertIn("state", out["error"])

    def test_requires_questions_or_preset(self):
        out = json.loads(tools.handle_decide({"state": "hello"}))
        self.assertFalse(out["success"])
        self.assertIn("questions", out["error"])

    def test_rejects_both_questions_and_preset(self):
        out = json.loads(tools.handle_decide({
            "state": "x", "preset": "triage",
            "questions": {"q": {"type": "noul"}},
        }))
        self.assertFalse(out["success"])

    def test_rejects_bad_question_type(self):
        out = json.loads(tools.handle_decide({
            "state": "x", "questions": {"q": {"type": "essay"}},
        }))
        self.assertFalse(out["success"])
        self.assertIn("essay", out["error"])

    def test_choice_requires_criteria(self):
        out = json.loads(tools.handle_decide({
            "state": "x", "questions": {"q": {"type": "choice", "instructions": "pick"}},
        }))
        self.assertFalse(out["success"])
        self.assertIn("criteria", out["error"])

    def test_success_path(self):
        with mock.patch.object(backend, "predict",
                               return_value=(_fake_result(), 12.3, "mlx", "multilingual")):
            out = json.loads(tools.handle_decide({
                "state": "I was billed twice",
                "questions": {"answer": {"type": "choice", "instructions": "dept?",
                                         "criteria": ["billing", "technical", "other"]}},
            }))
        self.assertTrue(out["success"])
        self.assertEqual(out["answers"]["answer"]["choice"], "billing")
        self.assertEqual(out["meta"]["backend"], "mlx")
        self.assertEqual(out["meta"]["latency_ms"], 12.3)
        self.assertEqual(out["action"]["act_probability"], 0.87)

    def test_preset_path(self):
        with mock.patch.object(backend, "get_preset",
                               return_value={"q": {"type": "noul", "instructions": "x"}}), \
             mock.patch.object(backend, "predict",
                               return_value=(_fake_result(), 1.0, "mlx", "multilingual")):
            out = json.loads(tools.handle_decide({"state": "ticket text", "preset": "triage"}))
        self.assertTrue(out["success"])

    def test_backend_unavailable_is_json_error(self):
        with mock.patch.object(backend, "predict",
                               side_effect=backend.BackendUnavailableError("install laya-mlx")):
            out = json.loads(tools.handle_decide({
                "state": "x", "questions": {"q": {"type": "noul"}},
            }))
        self.assertFalse(out["success"])
        self.assertIn("laya-mlx", out["error"])

    def test_never_raises(self):
        with mock.patch.object(backend, "predict", side_effect=RuntimeError("boom")):
            out = json.loads(tools.handle_decide({
                "state": "x", "questions": {"q": {"type": "noul"}},
            }))
        self.assertFalse(out["success"])
        self.assertIn("boom", out["error"])


class SlashTests(unittest.TestCase):
    def test_help(self):
        out = json.loads(tools.handle_slash("help"))
        self.assertTrue(out["success"])
        self.assertTrue(any("preset triage" in u for u in out["usage"]))

    def test_noul_form(self):
        captured = {}

        def fake_decide(args, **kw):
            captured.update(args)
            return json.dumps({"success": True})

        with mock.patch.object(tools, "handle_decide", side_effect=fake_decide):
            tools.handle_slash("noul; Refund requested?; I want my money back")
        self.assertEqual(captured["state"], "I want my money back")
        self.assertEqual(captured["questions"]["answer"]["type"], "noul")

    def test_choice_form(self):
        captured = {}

        def fake_decide(args, **kw):
            captured.update(args)
            return json.dumps({"success": True})

        with mock.patch.object(tools, "handle_decide", side_effect=fake_decide):
            tools.handle_slash("choice; Which dept?; billing, technical, other; billed twice")
        q = captured["questions"]["answer"]
        self.assertEqual(q["type"], "choice")
        self.assertEqual(q["criteria"], ["billing", "technical", "other"])
        self.assertEqual(captured["state"], "billed twice")

    def test_bad_form(self):
        out = json.loads(tools.handle_slash("essay; whatever"))
        self.assertFalse(out["success"])

    def test_choice_needs_two_options(self):
        out = json.loads(tools.handle_slash("choice; pick; onlyone; state text"))
        self.assertFalse(out["success"])


class BackendTests(unittest.TestCase):
    def test_detect_forced(self):
        with mock.patch.dict(os.environ, {"LAYA_BACKEND": "coreml"}):
            self.assertEqual(backend.detect_backend(), "coreml")

    def test_detect_auto_apple_silicon(self):
        env = {k: v for k, v in os.environ.items() if k != "LAYA_BACKEND"}
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(backend, "_is_apple_silicon", return_value=True), \
             mock.patch.object(backend, "_module_available",
                               side_effect=lambda b: b == "coreml"):
            self.assertEqual(backend.detect_backend(), "coreml")

    def test_unknown_model_alias(self):
        with self.assertRaises(ValueError):
            backend._checkpoint("nope", "mlx")


if __name__ == "__main__":
    unittest.main()
