"""记忆存储（长期 / 短期）单测。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from resume_agent.user_profile_store import (
    LongTermRecord,
    load_long_term,
    save_long_term,
    sanitize_user_id,
    user_storage_dir,
)
from resume_agent.schemas import ParsedResume


class TestSanitizeUserId(unittest.TestCase):
    def test_strips_and_default(self) -> None:
        self.assertEqual(sanitize_user_id("  u1  "), "u1")
        self.assertEqual(sanitize_user_id(""), "default")
        self.assertEqual(sanitize_user_id("   "), "default")

    def test_rejects_unsafe(self) -> None:
        with self.assertRaises(ValueError):
            sanitize_user_id("../etc")
        with self.assertRaises(ValueError):
            sanitize_user_id("a/b")
        with self.assertRaises(ValueError):
            sanitize_user_id(".")


class TestLongTermPersistence(unittest.TestCase):
    def test_roundtrip_under_custom_data_dir(self) -> None:
        root = Path(__file__).resolve().parent / "_mem_test_root"
        root.mkdir(exist_ok=True)
        try:
            with patch("resume_agent.user_profile_store.default_data_dir", return_value=root):
                uid = "testuser"
                raw = "张三\n求职：后端开发"
                parsed = ParsedResume(
                    name="张三",
                    target_role="后端开发",
                    skills=["Python"],
                )
                rec = LongTermRecord.from_parsed(raw, parsed)
                save_long_term(uid, rec)
                sub = user_storage_dir(uid)
                self.assertTrue((sub / "long_term.json").is_file())
                loaded = load_long_term(uid)
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertEqual(loaded.parsed_resume.get("name"), "张三")
                self.assertEqual(loaded.parsed_resume.get("target_role"), "后端开发")
                data = json.loads((sub / "long_term.json").read_text(encoding="utf-8"))
                self.assertIn("resume_fingerprint", data)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
