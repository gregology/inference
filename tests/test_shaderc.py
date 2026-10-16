"""Offline tests for the glslc build and its effect on the llama.cpp build id."""

import argparse
import unittest
from unittest import mock


def _step(cls):
    from installer.config import load_config
    from installer.hardware import Hardware
    ns = argparse.Namespace(
        models_toml=None, prune=False, dry_run=True, host="0.0.0.0", port=8000,
    )
    return cls(load_config(ns), Hardware())


class TestShadercCheck(unittest.TestCase):

    def _check(self, exists, version_output):
        from installer.steps import shaderc
        step = _step(shaderc.ShadercStep)
        with mock.patch.object(shaderc.Path, "exists", return_value=exists), \
                mock.patch.object(step, "sh_output", return_value=version_output):
            return step.check()

    def test_pinned_version_installed(self):
        from installer.steps.shaderc import SHADERC_TAG
        out = f"shaderc {SHADERC_TAG} {SHADERC_TAG}\nspirv-tools v2026.4 v2022.4-1347-gef96ed76\n"
        self.assertTrue(self._check(True, out))

    def test_other_version_installed(self):
        out = "shaderc v2023.2 v2023.2\nspirv-tools v2022.4\n"
        self.assertFalse(self._check(True, out))

    def test_dev_build_of_newer_tag_is_not_pinned(self):
        from installer.steps.shaderc import SHADERC_TAG
        self.assertFalse(self._check(True, f"shaderc {SHADERC_TAG}-dev abc123\n"))

    def test_missing(self):
        self.assertFalse(self._check(False, ""))


class TestBuildId(unittest.TestCase):

    def test_build_id_tracks_glslc(self):
        """Upgrading glslc must invalidate an existing llama.cpp build."""
        from installer.steps.build_llama import BuildLlamaStep
        from installer.steps.shaderc import SHADERC_TAG
        build_id = BuildLlamaStep._build_id("deadbeef")
        self.assertTrue(build_id.startswith("deadbeef "))
        self.assertIn(SHADERC_TAG, build_id)


if __name__ == "__main__":
    unittest.main()
