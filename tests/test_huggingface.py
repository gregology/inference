"""Validate that every model in models.toml exists on Hugging Face.

These tests hit the HF API so they require network access. Run with:
    python3 -m pytest tests/test_huggingface.py -v
or:
    python3 -m unittest tests.test_huggingface -v
"""

import argparse
import fnmatch
import json
import unittest
import urllib.error
import urllib.request

HF_API = "https://huggingface.co/api/models"


def _load():
    from installer.config import load_config
    ns = argparse.Namespace(
        models_toml=None, prune=False, dry_run=True, host="0.0.0.0", port=8000,
    )
    return load_config(ns)


def _get_repo_files(repo: str) -> list[str]:
    """Fetch the file listing for a HF repo. Returns list of relative paths."""
    url = f"{HF_API}/{repo}"
    req = urllib.request.Request(url, headers={"User-Agent": "inference-tests"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return [s["rfilename"] for s in data.get("siblings", [])]


class TestHuggingFaceManifest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = _load()
        cls._repo_cache: dict[str, list[str]] = {}

    def _files_for(self, repo: str) -> list[str]:
        if repo not in self._repo_cache:
            self._repo_cache[repo] = _get_repo_files(repo)
        return self._repo_cache[repo]

    def test_repos_exist(self):
        """Every repo in models.toml must be accessible on HF."""
        for model in self.config.models:
            with self.subTest(model=model.id, repo=model.repo):
                try:
                    files = self._files_for(model.repo)
                    self.assertIsInstance(files, list)
                except urllib.error.HTTPError as e:
                    self.fail(f"{model.repo}: HTTP {e.code}")

    def test_single_files_exist(self):
        """For models with file=, that exact file must exist in the repo."""
        for model in self.config.models:
            if not model.file:
                continue
            with self.subTest(model=model.id, file=model.file):
                files = self._files_for(model.repo)
                self.assertIn(
                    model.file, files,
                    f"{model.id}: '{model.file}' not found in {model.repo}. "
                    f"Available: {[f for f in files if f.endswith('.gguf')]}",
                )

    def test_include_patterns_match(self):
        """For models with include=, the pattern must match at least one file."""
        for model in self.config.models:
            if not model.include:
                continue
            with self.subTest(model=model.id, include=model.include):
                files = self._files_for(model.repo)
                matches = [f for f in files if fnmatch.fnmatch(f, model.include)]
                self.assertGreater(
                    len(matches), 0,
                    f"{model.id}: include pattern '{model.include}' matched no files in {model.repo}. "
                    f"Available: {[f for f in files if f.endswith('.gguf')]}",
                )


if __name__ == "__main__":
    unittest.main()
