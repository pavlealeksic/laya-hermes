"""Unit tests for the Laya context engine — stand-in base class, stubbed predict."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hermes_laya import engine as engine_mod  # noqa: E402
from hermes_laya.engine import LayaContextCompressor  # noqa: E402


def make_transcript(pairs=6, result_len=3000, tail=2):
    msgs = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Fix the flaky test in the parser."},
    ]
    for i in range(pairs):
        msgs.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "terminal", "arguments": '{"command": "ls"}'}}],
        })
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * result_len})
    msgs.append({"role": "user", "content": "did it work?"})
    msgs.append({"role": "assistant", "content": "checking now"})
    return msgs


def predict_with(keep_call, keep_result):
    return ({"answers": {"keep_call": {"noul": keep_call},
                         "keep_result": {"noul": keep_result}}}, 5.0, "mlx", "multilingual")


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = LayaContextCompressor()

    def _prune(self, msgs, tail_tokens=100000):
        return self.engine._prune_old_tool_results(
            msgs, protect_tail_count=2, protect_tail_tokens=tail_tokens)

    # -- decision mapping ---------------------------------------------------------

    def test_all_kept_falls_back_on_no_progress(self):
        with mock.patch.object(engine_mod.backend, "predict",
                               side_effect=lambda *a, **k: predict_with(0.99, 0.99)):
            out, n = self._prune(make_transcript())
        # nothing pruned -> reduction gate fails -> deterministic fallback
        self.assertEqual(self.engine.laya_fallbacks, 1)
        self.assertEqual(len(out), len(make_transcript()))

    def test_drop_result_truncates_with_marker(self):
        # keep_call high, keep_result low -> truncate result, keep pairing
        with mock.patch.object(engine_mod.backend, "predict",
                               side_effect=lambda *a, **k: predict_with(0.9, 0.1)):
            out, n = self._prune(make_transcript())
        self.assertGreater(n, 0)
        tool_rows = [m for m in out if m.get("role") == "tool"]
        self.assertTrue(all("[laya-compaction truncated" in m["content"] for m in tool_rows))
        self.assertTrue(engine_mod._valid_openai_sequence(out))

    def test_drop_call_removes_pair_and_stays_valid(self):
        with mock.patch.object(engine_mod.backend, "predict",
                               side_effect=lambda *a, **k: predict_with(0.05, 0.05)):
            out, n = self._prune(make_transcript())
        self.assertGreater(n, 0)
        self.assertFalse(any(m.get("role") == "tool" for m in out))
        self.assertTrue(engine_mod._valid_openai_sequence(out))

    def test_error_results_get_lower_keep_bar(self):
        # keep_call/keep_result between 0.25 and 0.5: normal units drop, error units keep
        msgs = make_transcript()
        msgs[3]["content"] = "Traceback: boom failed\n" + "x" * 3000  # c0 result = error

        def judge(state, questions, **kw):
            return predict_with(0.3, 0.3)

        with mock.patch.object(engine_mod.backend, "predict", side_effect=judge):
            out, n = self._prune(msgs)
        kept_texts = [m["content"] for m in out if m.get("role") == "tool"]
        self.assertTrue(any("Traceback" in t for t in kept_texts))  # error kept verbatim
        self.assertTrue(engine_mod._valid_openai_sequence(out))

    # -- gates and fallbacks ------------------------------------------------------

    def test_reduction_gate_falls_back(self):
        # one drop among eleven keeps -> reduction < 10% -> fallback
        calls = iter([predict_with(0.99, 0.99)] * 11 + [predict_with(0.0, 0.0)])
        with mock.patch.object(engine_mod.backend, "predict", side_effect=lambda *a, **k: next(calls)):
            out, n = self._prune(make_transcript(pairs=12, result_len=3000))
        self.assertEqual(self.engine.laya_fallbacks, 1)

    def test_predict_exception_falls_back(self):
        with mock.patch.object(engine_mod.backend, "predict", side_effect=RuntimeError("down")):
            out, n = self._prune(make_transcript())
        self.assertEqual(self.engine.laya_fallbacks, 1)
        self.assertEqual(len(out), len(make_transcript()))  # stand-in fallback = unchanged

    def test_proactive_path_bypasses_laya(self):
        with mock.patch.object(engine_mod.backend, "predict") as p:
            out, n = self.engine._prune_old_tool_results(
                make_transcript(), protect_tail_count=2, protect_tail_tokens=None)
            p.assert_not_called()

    def test_no_candidates_falls_back(self):
        msgs = make_transcript(result_len=100)  # below min_result_chars
        with mock.patch.object(engine_mod.backend, "predict") as p:
            out, n = self._prune(msgs)
            p.assert_not_called()
        self.assertEqual(self.engine.laya_fallbacks, 1)

    def test_bad_probability_falls_back(self):
        bad = ({"answers": {"keep_call": {"noul": 1.7}, "keep_result": {"noul": 0.5}}},
               1.0, "mlx", "multilingual")
        with mock.patch.object(engine_mod.backend, "predict", return_value=bad):
            out, n = self._prune(make_transcript())
        self.assertEqual(self.engine.laya_fallbacks, 1)

    # -- helpers ------------------------------------------------------------------

    def test_candidates_skip_duplicates_and_unpaired(self):
        msgs = make_transcript(pairs=2)
        # insert a duplicate result and an unpaired call BEFORE the protected tail
        dup = {"role": "tool", "tool_call_id": "c0", "content": "y" * 3000}
        lonely = {"role": "assistant", "content": "", "tool_calls": [
            {"id": "lonely", "type": "function",
             "function": {"name": "terminal", "arguments": "{}"}}]}
        msgs.insert(-2, dup)
        msgs.insert(-2, lonely)
        cands = engine_mod._collect_candidates(msgs, boundary=len(msgs) - 2, min_result_chars=2000)
        ids = {c.call_id for c in cands}
        self.assertNotIn("c0", ids)       # duplicated -> ambiguous -> skipped
        self.assertIn("c1", ids)
        self.assertNotIn("lonely", ids)   # unpaired

    def test_candidates_respect_boundary(self):
        msgs = make_transcript(pairs=4)
        cands = engine_mod._collect_candidates(msgs, boundary=6, min_result_chars=2000)
        self.assertTrue(all(c.result_idx < 6 for c in cands))

    def test_valid_sequence_rejects_orphans(self):
        self.assertFalse(engine_mod._valid_openai_sequence(
            [{"role": "tool", "tool_call_id": "ghost", "content": "x"}]))
        self.assertFalse(engine_mod._valid_openai_sequence([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        ]))  # call without result

    def test_state_fits_small_budget(self):
        msgs = make_transcript()
        cands = engine_mod._collect_candidates(msgs, boundary=len(msgs) - 2, min_result_chars=2000)
        state = self.engine._build_state(msgs, cands[0])
        import json
        self.assertLess(len(json.dumps(state)), 4000)  # ~1k tokens, well under 1024-token window

    def test_deepcopy(self):
        import copy
        self.engine.laya_calls = 7
        dup = copy.deepcopy(self.engine)
        self.assertEqual(dup.laya_calls, 7)
        self.assertEqual(dup.name, "laya")


if __name__ == "__main__":
    unittest.main()
