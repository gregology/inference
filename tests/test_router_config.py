"""Validate that the generated models.ini is well-formed."""

import argparse
import configparser
import re
import unittest


def _load():
    from installer.config import load_config
    ns = argparse.Namespace(
        models_toml=None, prune=False, dry_run=True, host="0.0.0.0", port=8000,
    )
    return load_config(ns)


def _generate_ini(config) -> str:
    from installer.steps.router_config import render_ini
    return render_ini(config)


def _parse_ini(content: str) -> configparser.ConfigParser:
    """Parse llama.cpp models.ini, skipping the preamble before the first section."""
    # Strip lines before the first [section] — llama.cpp puts 'version = 1' there.
    lines = content.splitlines(keepends=True)
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("["):
            start = i
            break
    trimmed = "".join(lines[start:])
    parser = configparser.ConfigParser()
    parser.read_string(trimmed)
    return parser


class TestRouterConfig(unittest.TestCase):

    def test_ini_is_parseable(self):
        """The generated ini must parse without errors."""
        config = _load()
        ini_content = _generate_ini(config)
        parser = _parse_ini(ini_content)
        expected_profiles = sum(len(m.profiles) for m in config.models)
        non_default = [s for s in parser.sections() if s != "DEFAULT"]
        self.assertEqual(len(non_default), expected_profiles + 1)  # +1 for [*]

    def test_every_profile_has_model_path(self):
        config = _load()
        ini_content = _generate_ini(config)
        parser = _parse_ini(ini_content)
        for section in parser.sections():
            if section == "*":
                continue
            with self.subTest(section=section):
                self.assertTrue(
                    parser.has_option(section, "model"),
                    f"[{section}] missing 'model' key",
                )
                model_path = parser.get(section, "model")
                self.assertTrue(
                    model_path.startswith("/srv/llm/models/"),
                    f"[{section}] model path doesn't start with /srv/llm/models/: {model_path}",
                )

    def test_has_version_line(self):
        config = _load()
        ini_content = _generate_ini(config)
        self.assertIn("version = 1", ini_content)

    def test_no_duplicate_sections(self):
        """Profile names must not collide — configparser silently merges dupes."""
        config = _load()
        ini_content = _generate_ini(config)
        sections = []
        for line in ini_content.splitlines():
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                sections.append(stripped[1:-1])
        self.assertEqual(len(sections), len(set(sections)), "Duplicate ini sections")

    def test_profiles_alias_their_own_name(self):
        """llama.cpp rewrites 'name:27b' sections to 'name:27B'; the alias keeps 'name:27b' routable."""
        config = _load()
        parser = _parse_ini(_generate_ini(config))
        for model in config.models:
            for profile in model.profiles:
                with self.subTest(profile=profile.name):
                    self.assertEqual(parser.get(profile.name, "alias"), profile.name)

    def test_profile_parallel_overrides_default(self):
        """A per-profile parallel lands in that profile's section, after the [*] default."""
        config = _load()
        parser = _parse_ini(_generate_ini(config))
        for model in config.models:
            for profile in model.profiles:
                if "parallel" in profile.extra:
                    with self.subTest(profile=profile.name):
                        self.assertEqual(parser.getint(profile.name, "parallel"), profile.extra["parallel"])

    def test_threads_batch_only_with_one_slot(self):
        """llama.cpp uses batch threads whenever slots generate together; on the O6 the extra A520 cores slow that down."""
        config = _load()
        for model in config.models:
            for profile in model.profiles:
                if "threads_batch" in profile.extra:
                    with self.subTest(profile=profile.name):
                        self.assertEqual(profile.extra.get("parallel", config.defaults.parallel), 1)

    def test_gpu_profiles_use_one_cpu_thread(self):
        """Idle CPU threads of a GPU model take cores from a CPU model running beside it."""
        config = _load()
        for model in config.models:
            for profile in model.profiles:
                if (profile.device or config.defaults.device) != "none":
                    with self.subTest(profile=profile.name):
                        self.assertEqual(profile.extra.get("threads"), 1)

    def test_mmap_profiles_skip_repack(self):
        """Repacked weights move out of the page cache into RAM, starving mmap'd experts."""
        config = _load()
        for model in config.models:
            for profile in model.profiles:
                if profile.extra.get("load_mode") == "mmap":
                    with self.subTest(profile=profile.name):
                        self.assertIs(profile.extra.get("no_repack"), True)

    def test_prompt_cache_enabled(self):
        """Prompt processing is slow on the O6; the RAM prompt cache must stay on."""
        config = _load()
        parser = _parse_ini(_generate_ini(config))
        self.assertNotEqual(parser.getint("*", "cache-ram"), 0)  # 0 disables it


if __name__ == "__main__":
    unittest.main()
