"""Offline tests for the llama-router systemd unit."""

import unittest


class TestRenderUnit(unittest.TestCase):

    def test_mali_env(self):
        """On Mali, coopmat is disabled and int8 mat-vec forced for the router."""
        from installer.steps.systemd import render_unit
        unit = render_unit("0.0.0.0", 8000, mali=True)
        self.assertIn("Environment=GGML_VK_DISABLE_COOPMAT=1\n", unit)
        self.assertIn("Environment=GGML_VK_FORCE_MMVQ=1\n", unit)

    def test_no_mali_env_elsewhere(self):
        """Other GPUs keep ggml-vulkan's defaults (coopmat helps on AMD/NVIDIA/Intel)."""
        from installer.steps.systemd import render_unit
        unit = render_unit("0.0.0.0", 8000)
        self.assertNotIn("GGML_VK_", unit)

    def test_no_per_model_flags(self):
        """Router flags override every preset; --parallel here would mask models.ini."""
        from installer.steps.systemd import render_unit
        self.assertNotIn("--parallel", render_unit("0.0.0.0", 8000, mali=True))

    def test_models_max(self):
        """The router flag comes from models.toml's [router] table."""
        from installer.steps.systemd import render_unit
        self.assertIn("--models-max 2 ", render_unit("0.0.0.0", 8000, models_max=2))

    def test_env_precedes_exec(self):
        from installer.steps.systemd import render_unit
        unit = render_unit("127.0.0.1", 9000, mali=True)
        self.assertLess(unit.index("GGML_VK_FORCE_MMVQ"), unit.index("ExecStart="))
        self.assertIn("--host 127.0.0.1 --port 9000", unit)


if __name__ == "__main__":
    unittest.main()
