"""Tests for backend auto-detection and configuration."""

import unittest

from installer.backend import (
    BackendConfig,
    BackendType,
    detect_backend,
    get_backend_config,
)
from installer.hardware import Hardware


class TestDetectBackend(unittest.TestCase):

    def test_nvidia_driver_selects_cuda(self):
        hw = Hardware(nvidia_driver_loaded=True, render_nodes=["/dev/dri/renderD128"])
        self.assertEqual(detect_backend(hw), BackendType.CUDA)

    def test_render_nodes_select_vulkan(self):
        hw = Hardware(render_nodes=["/dev/dri/renderD128"])
        self.assertEqual(detect_backend(hw), BackendType.VULKAN)

    def test_bare_machine_selects_cpu(self):
        hw = Hardware()
        self.assertEqual(detect_backend(hw), BackendType.CPU)

    def test_nvidia_takes_priority_over_vulkan(self):
        hw = Hardware(
            nvidia_driver_loaded=True,
            render_nodes=["/dev/dri/renderD128"],
            has_pcie_gpu=True,
            pcie_gpu_vendor="nvidia",
        )
        self.assertEqual(detect_backend(hw), BackendType.CUDA)


class TestGetBackendConfig(unittest.TestCase):

    def test_vulkan_config(self):
        hw = Hardware(gpu_device="Some GPU")
        cfg = get_backend_config(BackendType.VULKAN, hw)
        self.assertEqual(cfg.type, BackendType.VULKAN)
        self.assertEqual(cfg.default_device, "Vulkan0")
        self.assertEqual(cfg.build_dir_name, "build-vulkan")
        self.assertIn("-DGGML_VULKAN", cfg.cmake_flags)
        self.assertIn("libvulkan-dev", cfg.packages)
        self.assertNotIn("vulkan-headers", cfg.skip_steps)
        self.assertIsNone(cfg.llama_cpp_ref_override)

    def test_cuda_config(self):
        hw = Hardware(nvidia_driver_loaded=True)
        cfg = get_backend_config(BackendType.CUDA, hw)
        self.assertEqual(cfg.type, BackendType.CUDA)
        self.assertEqual(cfg.default_device, "CUDA0")
        self.assertEqual(cfg.build_dir_name, "build-cuda")
        self.assertIn("-DGGML_CUDA", cfg.cmake_flags)
        self.assertIn("vulkan-headers", cfg.skip_steps)

    def test_cpu_config(self):
        hw = Hardware()
        cfg = get_backend_config(BackendType.CPU, hw)
        self.assertEqual(cfg.type, BackendType.CPU)
        self.assertEqual(cfg.default_device, "none")
        self.assertEqual(cfg.build_dir_name, "build-cpu")
        self.assertEqual(cfg.cmake_flags, {})
        self.assertIn("vulkan-headers", cfg.skip_steps)
        self.assertIn("gpu-permissions", cfg.skip_steps)

    def test_mali_gpu_pins_llama_cpp(self):
        hw = Hardware(gpu_device="Arm Mali-G720-Immortalis MC12")
        cfg = get_backend_config(BackendType.VULKAN, hw)
        self.assertEqual(cfg.llama_cpp_ref_override, "914dde72b")

    def test_non_mali_vulkan_no_pin(self):
        hw = Hardware(gpu_device="AMD Radeon RX 7900 XTX")
        cfg = get_backend_config(BackendType.VULKAN, hw)
        self.assertIsNone(cfg.llama_cpp_ref_override)

    def test_immortalis_detected_as_mali(self):
        hw = Hardware(gpu_device="Immortalis G720")
        cfg = get_backend_config(BackendType.VULKAN, hw)
        self.assertEqual(cfg.llama_cpp_ref_override, "914dde72b")


if __name__ == "__main__":
    unittest.main()
