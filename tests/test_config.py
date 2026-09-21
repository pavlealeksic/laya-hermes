"""Unit tests for the settings layer (config.py) and its slash-command surface."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
import tools  # noqa: E402


class FakeCtx:
    def __init__(self, settings=None):
        self.settings = dict(settings or {})
        self.writes = []

    def get_config(self, key, default=None):
        return self.settings.get(key, default)

    def set_config(self, key, value):
        self.writes.append((key, value))
        self.settings[key] = value


class ConfigTests(unittest.TestCase):
    def tearDown(self):
        config.init(None)

    def test_default_when_unset(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LAYA_")}
        with mock.patch.dict(os.environ, env, clear=True):
            value, source = config.get("filter_output")
        self.assertEqual((value, source), (False, "default"))

    def test_env_override_and_coercion(self):
        with mock.patch.dict(os.environ, {"LAYA_FILTER_OUTPUT": "1", "LAYA_FILTER_MIN_CHARS": "9000"}):
            self.assertEqual(config.get("filter_output"), (True, "env"))
            self.assertEqual(config.get("filter_min_chars"), (9000, "env"))

    def test_settings_used_when_no_env(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LAYA_")}
        config.init(FakeCtx({"routing_hint": True, "filter_min_chars": 3000}))
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(config.get("routing_hint"), (True, "settings"))
            self.assertEqual(config.get("filter_min_chars"), (3000, "settings"))

    def test_env_beats_settings(self):
        config.init(FakeCtx({"routing_hint": False}))
        with mock.patch.dict(os.environ, {"LAYA_ROUTING_HINT": "true"}):
            self.assertEqual(config.get("routing_hint"), (True, "env"))

    def test_set_value_persists_via_ctx(self):
        ctx = FakeCtx()
        config.init(ctx)
        value, message = config.set_value("filter_output", "true")
        self.assertTrue(value)
        self.assertEqual(ctx.writes, [("filter_output", True)])
        self.assertIn("immediately", message)

    def test_set_value_int(self):
        ctx = FakeCtx()
        config.init(ctx)
        config.set_value("filter_min_chars", "4500")
        self.assertEqual(ctx.writes, [("filter_min_chars", 4500)])

    def test_set_value_rejects_unknown_key(self):
        config.init(FakeCtx())
        with self.assertRaises(KeyError):
            config.set_value("nope", "1")

    def test_set_value_rejects_bad_bool(self):
        config.init(FakeCtx())
        with self.assertRaises(ValueError):
            config.set_value("filter_output", "maybe")

    def test_set_value_requires_ctx(self):
        with self.assertRaises(RuntimeError):
            config.set_value("filter_output", "true")

    def test_describe_covers_all_settings(self):
        keys = {entry["key"] for entry in config.describe()}
        self.assertEqual(keys, set(config.SETTINGS))


class SlashConfigTests(unittest.TestCase):
    def tearDown(self):
        config.init(None)

    def test_config_lists_settings(self):
        out = json.loads(tools.handle_slash("config"))
        self.assertTrue(out["success"])
        keys = {s["key"] for s in out["settings"]}
        self.assertIn("filter_output", keys)
        self.assertIn("backend", keys)

    def test_set_form(self):
        config.init(FakeCtx())
        out = json.loads(tools.handle_slash("set routing_hint on"))
        self.assertTrue(out["success"])
        self.assertEqual(out["value"], True)

    def test_set_form_bad_key(self):
        config.init(FakeCtx())
        out = json.loads(tools.handle_slash("set bogus 1"))
        self.assertFalse(out["success"])

    def test_set_form_usage_error(self):
        out = json.loads(tools.handle_slash("set"))
        self.assertFalse(out["success"])
        self.assertIn("Usage", out["error"])

    def test_help_lists_config_forms(self):
        out = json.loads(tools.handle_slash("help"))
        self.assertTrue(any("config" in u for u in out["usage"]))
        self.assertTrue(any("set " in u for u in out["usage"]))


if __name__ == "__main__":
    unittest.main()
