"""Offline tests for models.toml parsing and validation."""

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load():
    from installer.config import load_config
    ns = argparse.Namespace(
        models_toml=None, prune=False, dry_run=True, host="0.0.0.0", port=8000,
    )
    return load_config(ns)


class TestConfigLoading(unittest.TestCase):

    def test_loads_without_error(self):
        config = _load()
        self.assertGreater(len(config.models), 0)

    def test_every_model_has_source(self):
        """Every model must specify either file or include."""
        config = _load()
        for model in config.models:
            with self.subTest(model=model.id):
                self.assertTrue(
                    model.file or model.include,
                    f"{model.id}: must have 'file' or 'include'",
                )

    def test_every_model_has_at_least_one_profile(self):
        config = _load()
        for model in config.models:
            with self.subTest(model=model.id):
                self.assertGreater(
                    len(model.profiles), 0,
                    f"{model.id}: no profiles defined",
                )

    def test_no_duplicate_profile_names(self):
        config = _load()
        names = []
        for model in config.models:
            for profile in model.profiles:
                names.append(profile.name)
        self.assertEqual(len(names), len(set(names)), f"Duplicate profile names: {names}")

    def test_large_models_use_include(self):
        """Large models should use include (multi-shard), not a single file."""
        config = _load()
        for model in config.models:
            if model.large:
                with self.subTest(model=model.id):
                    self.assertIsNotNone(
                        model.include,
                        f"{model.id}: large=true but no include pattern",
                    )

    def test_local_dirs_are_unique(self):
        config = _load()
        dirs = [m.local_dir for m in config.models]
        self.assertEqual(len(dirs), len(set(dirs)), f"Duplicate local_dir values")

    def test_profile_ctx_size_positive(self):
        config = _load()
        for model in config.models:
            for profile in model.profiles:
                with self.subTest(profile=profile.name):
                    self.assertGreater(profile.ctx_size, 0)

    def test_no_removed_llama_options(self):
        """llama.cpp removed these flags (#28334); the router rejects them."""
        removed = {"mmap", "no_mmap", "mlock", "no_mlock", "direct_io", "no_direct_io"}
        config = _load()
        for model in config.models:
            for profile in model.profiles:
                with self.subTest(profile=profile.name):
                    self.assertFalse(removed & set(profile.extra))


class TestFirstGguf(unittest.TestCase):

    def _model(self, tmp, **kwargs):
        from installer.config import Model
        model = Model(id="m", repo="org/repo", **kwargs)
        patcher = mock.patch.object(
            Model, "local_dir", new_callable=mock.PropertyMock, return_value=tmp,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return model

    def _touch(self, root, *names):
        for name in names:
            path = Path(root) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

    def test_single_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self._model(tmp, file="a.gguf")
            self.assertEqual(model.first_gguf, f"{tmp}/a.gguf")

    def test_shard_directory_include(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._touch(tmp, "Q4/x-00002-of-00002.gguf", "Q4/x-00001-of-00002.gguf",
                        "Q8/x-00001-of-00002.gguf")
            model = self._model(tmp, include="Q4/*")
            self.assertEqual(model.first_gguf, f"{tmp}/Q4/x-00001-of-00002.gguf")

    def test_root_level_include(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._touch(tmp, "x-IQ4_XS-00002-of-00002.gguf", "x-IQ4_XS-00001-of-00002.gguf",
                        "x-Q4_K_M-00001-of-00002.gguf", "mmproj-x-F16.gguf")
            model = self._model(tmp, include="x-IQ4_XS-*.gguf")
            self.assertEqual(model.first_gguf, f"{tmp}/x-IQ4_XS-00001-of-00002.gguf")

    def test_not_downloaded_falls_back_to_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._model(tmp, include="Q4/*").first_gguf, f"{tmp}/Q4/")
            self.assertEqual(self._model(tmp, include="x-*.gguf").first_gguf, f"{tmp}/")


if __name__ == "__main__":
    unittest.main()
