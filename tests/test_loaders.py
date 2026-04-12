"""简历文件加载逻辑单测。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from resume_agent.loaders import ResumeLoadError, load_resume_text


class TestLoadResumeText(unittest.TestCase):
    def test_loads_txt_utf8(self) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write("姓名：测\n技能：Python\n")
            path = Path(f.name)
        try:
            text = load_resume_text(path)
            self.assertIn("Python", text)
        finally:
            path.unlink(missing_ok=True)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(ResumeLoadError):
            load_resume_text(Path("/nonexistent/resume.txt"))

    def test_unsupported_suffix_raises(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(b"x")
            path = Path(f.name)
        try:
            with self.assertRaises(ResumeLoadError):
                load_resume_text(path)
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
