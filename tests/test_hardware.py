"""Offline tests for hardware detection."""

import unittest

# /proc/cpuinfo on the Radxa Orion O6 (CIX P1: Cortex-A720 + Cortex-A520).
O6_CPUINFO = """\
processor\t: 0
BogoMIPS\t: 2000.00
Features\t: fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdhp cpuid asimdrdm jscvt fcma lrcpc dcpop sha3 sm3 sm4 asimddp sha512 sve asimdfhm dit uscat ilrcpc flagm sb paca pacg dcpodp sve2 sveaes svepmull svebitperm svesha3 svesm4 flagm2 frint svei8mm svebf16 i8mm bf16 dgh bti mte ecv afp mte3 wfxt
CPU implementer\t: 0x41
CPU part\t: 0xd81
"""

# Raspberry Pi 4 (Cortex-A72, Armv8.0).
PI4_CPUINFO = "processor\t: 0\nFeatures\t: fp asimd evtstrm crc32 cpuid\nCPU part\t: 0xd08\n"

X86_CPUINFO = "processor\t: 0\nflags\t\t: fpu vme de pse tsc msr pae mce avx2\n"


class TestArmMarch(unittest.TestCase):

    def test_o6(self):
        """GCC 12 doesn't know the A720, so the extensions must be named."""
        from installer.hardware import arm_march
        self.assertEqual(arm_march(O6_CPUINFO), "armv8.2-a+fp16+dotprod+i8mm+bf16")

    def test_core_without_extensions(self):
        from installer.hardware import arm_march
        self.assertEqual(arm_march(PI4_CPUINFO), "armv8-a")

    def test_not_arm(self):
        """x86 keeps GGML_NATIVE, which detects its features correctly."""
        from installer.hardware import arm_march
        self.assertIsNone(arm_march(X86_CPUINFO))


class TestCpuFlags(unittest.TestCase):

    def _step(self, cpu_arch):
        import argparse
        from installer.config import load_config
        from installer.hardware import Hardware
        from installer.steps.build_llama import BuildLlamaStep
        ns = argparse.Namespace(
            models_toml=None, prune=False, dry_run=True, host="0.0.0.0", port=8000,
        )
        return BuildLlamaStep(load_config(ns), Hardware(cpu_arch=cpu_arch))

    def test_arm_target_replaces_native(self):
        flags = self._step("armv8.2-a+dotprod")._cpu_flags()
        self.assertIn("-DGGML_NATIVE=OFF", flags)
        self.assertIn("-DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod", flags)

    def test_native_elsewhere(self):
        self.assertEqual(self._step(None)._cpu_flags(), "")

    def test_build_id_tracks_cpu_target(self):
        """Builds from before the explicit target must be rebuilt."""
        from installer.steps.build_llama import BuildLlamaStep
        self.assertNotEqual(
            BuildLlamaStep._build_id("deadbeef"),
            BuildLlamaStep._build_id("deadbeef", "armv8.2-a+dotprod"),
        )


if __name__ == "__main__":
    unittest.main()
