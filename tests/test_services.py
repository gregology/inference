"""Offline tests for [services] in models.toml and their systemd units."""

import argparse
import tempfile
import unittest
from pathlib import Path


def _load(models_toml=None):
    from installer.config import load_config
    ns = argparse.Namespace(
        models_toml=models_toml, prune=False, dry_run=True, host="0.0.0.0", port=8000,
    )
    return load_config(ns)


def _service(**kwargs):
    from installer.config import Service
    kwargs.setdefault("package", "pkg==1.0")
    kwargs.setdefault("port", 10300)
    return Service(name="stt", **kwargs)


class TestServicesConfig(unittest.TestCase):

    def test_packages_are_pinned(self):
        """An unpinned package would change under a re-run; the check() marker relies on the spec."""
        for service in _load().services:
            with self.subTest(service=service.name):
                self.assertIn("==", service.package)

    def test_scripts_exist(self):
        config = _load()
        for service in config.services:
            if service.script:
                with self.subTest(service=service.name):
                    self.assertTrue((config.repo_dir / service.script).is_file())

    def test_services_listen_on_their_port(self):
        from installer.steps.services import exec_args
        for service in _load().services:
            with self.subTest(service=service.name):
                self.assertTrue(any(str(service.port) in a for a in exec_args(service)))

    def _toml(self, body: str) -> str:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False)
        tmp.write(body)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return tmp.name

    def test_port_collision_rejected(self):
        path = self._toml('[services.a]\npackage = "a==1"\nmodule = "a"\nport = 8000\n')
        with self.assertRaises(SystemExit):
            _load(path)

    def test_module_or_script_required(self):
        path = self._toml('[services.a]\npackage = "a==1"\nport = 9000\n')
        with self.assertRaises(SystemExit):
            _load(path)


class TestServiceUnits(unittest.TestCase):

    def test_placeholders(self):
        from installer.steps.services import exec_args
        service = _service(module="m", download="org/model-x",
                           args=["--uri", "tcp://0.0.0.0:{port}", "--model", "{download}", "--data-dir", "{data}"])
        self.assertEqual(exec_args(service), [
            "/srv/llm/services/stt/venv/bin/python", "-m", "m",
            "--uri", "tcp://0.0.0.0:10300",
            "--model", "/srv/llm/services/stt/data/model-x",
            "--data-dir", "/srv/llm/services/stt/data",
        ])

    def test_script_runs_from_service_dir(self):
        """The llm user can't read every repo checkout, so scripts are copied next to the venv."""
        from installer.steps.services import exec_args
        args = exec_args(_service(script="services/tool.py"))
        self.assertEqual(args[1], "/srv/llm/services/stt/tool.py")

    def test_quoting(self):
        from installer.steps.services import _systemd_quote
        self.assertEqual(_systemd_quote("plain"), "plain")
        self.assertEqual(_systemd_quote("two words"), '"two words"')
        self.assertEqual(_systemd_quote("50%"), "50%%")
        self.assertEqual(_systemd_quote("$HOME"), "$$HOME")

    def test_options_grouped_with_values(self):
        from installer.steps.services import render_unit
        unit = render_unit(_service(module="m", args=["--uri", "tcp://0.0.0.0:{port}", "--debug"]))
        self.assertIn("  -m m \\\n", unit)
        self.assertIn("  --uri tcp://0.0.0.0:10300 \\\n", unit)
        self.assertIn("  --debug\n", unit)

    def test_env_file(self):
        from installer.steps.services import MARKER, render_unit
        unit = render_unit(_service(module="m", env_file="/srv/llm/etc/key.env"))
        self.assertTrue(unit.startswith(MARKER))
        self.assertIn("EnvironmentFile=/srv/llm/etc/key.env\n", unit)
        self.assertLess(unit.index("EnvironmentFile="), unit.index("ExecStart="))
        self.assertNotIn("EnvironmentFile", render_unit(_service(module="m")))


class TestSearchFormatting(unittest.TestCase):

    def _mod(self):
        import importlib.util
        try:
            import mcp  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("mcp not installed")
        spec = importlib.util.spec_from_file_location(
            "deepseek_search_mcp", Path(__file__).resolve().parent.parent / "services/deepseek_search_mcp.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_hex_site_names_decoded(self):
        mod = self._mod()
        self.assertEqual(mod._site_name("6e686c652e636f6d"), "nhle.com")
        self.assertEqual(mod._site_name("USA Today"), "USA Today")

    def test_results_compact(self):
        mod = self._mod()
        text = mod.format_results({"results": [
            {"title": "T", "url": "https://x", "date": "2026-06-14T16:00:00+00:00",
             "site_name": "nhl.com", "summary": "word " * 200},
        ]})
        self.assertIn("1. T (2026-06-14, nhl.com)", text)
        self.assertLess(len(text), mod.SUMMARY_CHARS + 100)
        self.assertEqual(mod.format_results({"results": []}), "No results.")
        self.assertEqual(mod.format_results({"error": "DeepSeek search returned HTTP 429"}),
                         "DeepSeek search returned HTTP 429")


if __name__ == "__main__":
    unittest.main()
