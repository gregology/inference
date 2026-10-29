"""Offline tests for resident profiles (always-on llama-servers)."""

import argparse
import unittest


def _load():
    from installer.config import load_config
    ns = argparse.Namespace(
        models_toml=None, prune=False, dry_run=True, host="0.0.0.0", port=8000,
    )
    return load_config(ns)


def _resident():
    config = _load()
    for model in config.models:
        for profile in model.profiles:
            if profile.resident:
                yield config, model, profile


class TestServerArgs(unittest.TestCase):

    def _args(self, extra=None, **profile_kwargs):
        from installer.config import Defaults, Model, Profile
        from installer.steps.resident import server_args
        config = argparse.Namespace(defaults=Defaults(), host="0.0.0.0")
        model = Model(id="m", repo="org/repo", file="m.gguf")
        profile = Profile(name="voice:9b", port=8001, extra=extra or {}, **profile_kwargs)
        return server_args(config, model, profile)

    def test_model_and_alias(self):
        args = self._args()
        self.assertEqual(args[0], "--model /srv/llm/models/m/m.gguf")
        self.assertIn("--alias voice:9b", args)

    def test_defaults_apply(self):
        """A standalone server has no [*] section to inherit, so defaults go on its command line."""
        args = self._args()
        self.assertIn("--cache-ram 8192", args)
        self.assertIn("--cache-type-k q4_0", args)
        self.assertIn("--no-warmup", args)

    def test_profile_overrides_defaults(self):
        args = self._args(device="none", gpu_layers=0, ctx_size=32768)
        self.assertIn("--device none", args)
        self.assertNotIn("--device Vulkan0", args)
        self.assertIn("--n-gpu-layers 0", args)
        self.assertIn("--ctx-size 32768", args)

    def test_extra_keys_become_flags(self):
        args = self._args(extra={"threads_batch": 12, "jinja": True, "no_repack": False})
        self.assertIn("--threads-batch 12", args)
        self.assertIn("--jinja", args)
        self.assertFalse(any("no-repack" in a for a in args))

    def test_extra_overrides_defaults(self):
        args = self._args(extra={"parallel": 2})
        self.assertIn("--parallel 2", args)
        self.assertNotIn("--parallel 1", args)


class TestResidentUnits(unittest.TestCase):

    def test_unit_name(self):
        from installer.config import Profile
        from installer.steps.resident import unit_name
        self.assertEqual(unit_name(Profile(name="voice:9b", port=8001)), "llama-voice-9b.service")

    def test_unit_names_unique(self):
        from installer.steps.resident import unit_name
        names = [unit_name(p) for _, _, p in _resident()]
        self.assertEqual(len(names), len(set(names)))

    def test_unit_listens_on_profile_port(self):
        from installer.steps.resident import MARKER, render_unit
        for config, model, profile in _resident():
            with self.subTest(profile=profile.name):
                unit = render_unit(config, model, profile)
                self.assertTrue(unit.startswith(MARKER))
                self.assertIn(f"--host 0.0.0.0 --port {profile.port}", unit)
                self.assertNotIn("--models-preset", unit)

    def test_router_unit_is_not_managed_as_resident(self):
        """Stale-unit cleanup keys off the marker; the router unit must never carry it."""
        from installer.steps.resident import MARKER
        from installer.steps.systemd import render_unit
        self.assertNotIn(MARKER, render_unit("0.0.0.0", 8000, mali=True))

    def test_resident_profiles_keep_prompt_cache(self):
        """Resident models answer interactive clients; re-processing their system prompt costs seconds."""
        for config, model, profile in _resident():
            with self.subTest(profile=profile.name):
                self.assertNotEqual(config.defaults.cache_ram, 0)


if __name__ == "__main__":
    unittest.main()
