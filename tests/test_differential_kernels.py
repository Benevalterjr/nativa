#!/usr/bin/env python3
"""
Differential Verification Test Suite for Nativa Engine.
Verifies:
  1. Numerical equivalence of Q4x4 dot products, extreme values, zero blocks,
     BF16 scales, and embedding lookups between SIMD and Scalar backends.
  2. End-to-end benchmark comparison between SIMD and --scalar mode on real models.
"""

import os
import subprocess
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.compilar import compile_c


class TestDifferentialKernels(unittest.TestCase):
    """Tests numerical equivalence between SIMD and pure scalar kernels."""

    def test_q4x4_simd_vs_scalar_differential(self):
        """Compile and run the C99 differential verification suite."""
        src = os.path.join(ROOT_DIR, "tools", "teste_differential_q4x4.c")
        exe = os.path.join(ROOT_DIR, "tools", "temp_diff_test.exe" if sys.platform.startswith("win") else "tools/temp_diff_test")

        self.assertTrue(os.path.isfile(src), f"Source file {src} not found.")

        # Compile differential test binary
        ok = compile_c(src, exe)
        self.assertTrue(ok, "Failed to compile teste_differential_q4x4.c")

        try:
            # Run the differential test suite
            proc = subprocess.run([exe], capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, f"Differential test failed:\n{proc.stdout}\n{proc.stderr}")
            self.assertIn("DIFFERENTIAL TEST RESULT: 7 / 7 PASSED", proc.stdout)
        finally:
            if os.path.isfile(exe):
                try:
                    os.remove(exe)
                except OSError:
                    pass

    def test_e2e_simd_vs_scalar_hash_parity(self):
        """Verify that running nativa_engine with --scalar produces identical outputs."""
        engine_exe = "nativa_engine.exe" if sys.platform.startswith("win") else "nativa_engine"
        real_engine = os.path.join(ROOT_DIR, "tools", engine_exe)
        real_model = os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4")
        real_tok = os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json")

        if not (os.path.isfile(real_engine) and os.path.isfile(real_model) and os.path.isfile(real_tok)):
            self.skipTest("Real artifacts (nativa_engine, .natq4 model, or tokenizer) not found.")

        # 1. Run default (SIMD)
        cmd_simd = [real_engine, real_model, "--bench", "5"]
        proc_simd = subprocess.run(cmd_simd, capture_output=True, text=True, timeout=90)
        self.assertEqual(proc_simd.returncode, 0, f"SIMD bench failed: {proc_simd.stderr}")

        # 2. Run with --scalar
        cmd_scalar = [real_engine, real_model, "--bench", "5", "--scalar"]
        proc_scalar = subprocess.run(cmd_scalar, capture_output=True, text=True, timeout=90)
        self.assertEqual(proc_scalar.returncode, 0, f"Scalar bench failed: {proc_scalar.stderr}")

        # Check scalar mode announcement
        self.assertIn("C99 scalar backend forced!", proc_scalar.stdout)

        # Extract BENCH_HASH
        hash_simd = None
        hash_scalar = None
        for line in proc_simd.stdout.splitlines():
            if "BENCH_HASH=" in line:
                hash_simd = line.split("BENCH_HASH=")[1].strip()
        for line in proc_scalar.stdout.splitlines():
            if "BENCH_HASH=" in line:
                hash_scalar = line.split("BENCH_HASH=")[1].strip()

        self.assertIsNotNone(hash_simd, "SIMD benchmark did not produce BENCH_HASH.")
        self.assertIsNotNone(hash_scalar, "Scalar benchmark did not produce BENCH_HASH.")
        self.assertEqual(hash_simd, hash_scalar, f"Deterministic hash mismatch! SIMD: {hash_simd}, Scalar: {hash_scalar}")


if __name__ == "__main__":
    unittest.main()
