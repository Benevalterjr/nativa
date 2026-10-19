#!/usr/bin/env python3
"""
Regression Test Suite: Nativa Bundle (llamafile-style self-contained executables).
Covers:
  1. Constants, alignment (64 KB), and fixed trailer size (64 bytes).
  2. Packing, unpacking, and inspection of trailer metadata.
  3. Binary vocabulary export and serialization.
  4. Creation and validation of synthetic bundles (no real model required).
  5. Functional E2E test with a real model and native execution (if available in the environment).
"""

import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def safe_unlink(filepath: str, max_retries: int = 30, delay: float = 0.5):
    """Remove file ensuring process handles have been released by the OS on Windows."""
    for _ in range(max_retries):
        if not os.path.isfile(filepath):
            return
        try:
            os.unlink(filepath)
            return
        except PermissionError:
            time.sleep(delay)
        except OSError:
            break

from tools.nativa_bundle import (
    NATIVA_BUNDLE_ALIGN,
    NATIVA_BUNDLE_MAGIC,
    NATIVA_BUNDLE_VERSION,
    TRAILER_FORMAT,
    TRAILER_SIZE,
    align_up,
    create_bundle,
    export_vocab_bytes,
    inspect_bundle,
    write_vocab_file,
)


class TestNativaBundleUnit(unittest.TestCase):
    """Unit tests for trailer format, alignment, and vocabulary."""

    def test_trailer_constants_and_size(self):
        """The trailer must be exactly 64 bytes for alignment and binary consistency."""
        self.assertEqual(TRAILER_SIZE, 64, "NativaBundleTrailer size must be exactly 64 bytes.")
        self.assertEqual(NATIVA_BUNDLE_MAGIC, 0x4E4154495641464C, "Incorrect magic value.")
        self.assertEqual(NATIVA_BUNDLE_VERSION, 1, "Default trailer version must be 1.")
        self.assertEqual(NATIVA_BUNDLE_ALIGN, 65536, "Mandatory 64 KB alignment (MapViewOfFile / mmap).")

    def test_align_up(self):
        """Verify rounding of offsets to alignment multiples."""
        self.assertEqual(align_up(0, 65536), 0)
        self.assertEqual(align_up(1, 65536), 65536)
        self.assertEqual(align_up(65535, 65536), 65536)
        self.assertEqual(align_up(65536, 65536), 65536)
        self.assertEqual(align_up(65537, 65536), 131072)
        self.assertEqual(align_up(131072, 65536), 131072)
        # Test generic alignment
        self.assertEqual(align_up(15, 16), 16)
        self.assertEqual(align_up(16, 16), 16)
        self.assertEqual(align_up(17, 16), 32)

    def test_trailer_pack_unpack_roundtrip(self):
        """Verify that 64B packing and unpacking preserve all fields intact."""
        engine_sz = 1200000
        vocab_off = 1245184
        vocab_sz = 850000
        model_off = 2162688
        model_sz = 350000000
        version = 1
        flags = 0
        reserved = b"\x00" * 8

        packed = struct.pack(
            TRAILER_FORMAT,
            NATIVA_BUNDLE_MAGIC,
            engine_sz,
            vocab_off,
            vocab_sz,
            model_off,
            model_sz,
            version,
            flags,
            reserved,
        )
        self.assertEqual(len(packed), 64)

        unpacked = struct.unpack(TRAILER_FORMAT, packed)
        self.assertEqual(unpacked[0], NATIVA_BUNDLE_MAGIC)
        self.assertEqual(unpacked[1], engine_sz)
        self.assertEqual(unpacked[2], vocab_off)
        self.assertEqual(unpacked[3], vocab_sz)
        self.assertEqual(unpacked[4], model_off)
        self.assertEqual(unpacked[5], model_sz)
        self.assertEqual(unpacked[6], version)
        self.assertEqual(unpacked[7], flags)
        self.assertEqual(unpacked[8], reserved)

    def test_inspect_bundle_invalid_cases(self):
        """inspect_bundle must reject non-existent, truncated, or corrupted files."""
        # Non-existent file
        self.assertIsNone(inspect_bundle("caminho_inexistente_bundle_xyz.exe"))

        with tempfile.TemporaryDirectory() as tmpdir:
            # File smaller than 64 bytes
            too_small = os.path.join(tmpdir, "small.bin")
            with open(too_small, "wb") as f:
                f.write(b"NATIVA")
            self.assertIsNone(inspect_bundle(too_small))

            # 64-byte file with wrong magic
            bad_magic = os.path.join(tmpdir, "bad_magic.bin")
            with open(bad_magic, "wb") as f:
                f.write(b"\xFF" * 64)
            self.assertIsNone(inspect_bundle(bad_magic))

            # File with valid magic but incompatible version
            bad_version = os.path.join(tmpdir, "bad_version.bin")
            corrupt_trailer = struct.pack(
                TRAILER_FORMAT,
                NATIVA_BUNDLE_MAGIC,
                1000, 65536, 100, 131072, 200,
                999,  # Non-existent version
                0,
                b"\x00" * 8,
            )
            with open(bad_version, "wb") as f:
                f.write(b"\x00" * 131072)
                f.write(corrupt_trailer)
            self.assertIsNone(inspect_bundle(bad_version))

    def test_export_vocab_bytes(self):
        """Verify binary vocabulary serialization in the C engine format."""
        class MockTokenizer:
            def __init__(self):
                self.inv_vocab = {
                    0: "<|im_start|>",
                    1: "<|im_end|>",
                    2: "ola",
                    3: "<0x0A>",  # Newline byte
                }

        tok = MockTokenizer()
        data = export_vocab_bytes(tok)
        self.assertGreater(len(data), 4)

        # First uint32: total tokens
        vocab_size = struct.unpack_from("<I", data, 0)[0]
        self.assertEqual(vocab_size, 4)

        # Read tokens
        offset = 4
        tokens_read = []
        for _ in range(vocab_size):
            tok_len = struct.unpack_from("<H", data, offset)[0]
            offset += 2
            tok_bytes = data[offset : offset + tok_len]
            offset += tok_len
            tokens_read.append(tok_bytes)

        self.assertEqual(tokens_read[0], b"<|im_start|>")
        self.assertEqual(tokens_read[1], b"<|im_end|>")
        self.assertEqual(tokens_read[2], b"ola")
        self.assertEqual(tokens_read[3], b"\x0A")  # <0x0A> decoded to byte 10


class TestNativaBundleIntegration(unittest.TestCase):
    """Integration tests creating synthetic bundles and validating layout and alignment."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="nativa_bundle_test_")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_and_inspect_synthetic_bundle(self):
        """Create a synthetic bundle and validate 64 KB aligned offsets."""
        # 1. Create a mock engine
        fake_engine = os.path.join(self.tmpdir, "fake_engine.exe")
        engine_size = 95432  # Unaligned size
        with open(fake_engine, "wb") as f:
            f.write(b"MOCK_ENGINE" * (engine_size // 11) + b"END")
        real_engine_sz = os.path.getsize(fake_engine)

        # 2. Create a mock model
        fake_model = os.path.join(self.tmpdir, "fake_model.natq4")
        model_size = 200000  # Arbitrary size
        with open(fake_model, "wb") as f:
            # NATQ4 header
            f.write(b"NATQ4X4\x00")
            f.write(b"\x00" * (model_size - 8))

        # 3. Create a mock tokenizer JSON
        fake_tok_path = os.path.join(self.tmpdir, "fake_tok.json")
        tok_data = {
            "model": {
                "vocab": {
                    "<unk>": 0,
                    "bos": 1,
                    "eos": 2,
                    "teste": 3,
                }
            }
        }
        with open(fake_tok_path, "w", encoding="utf-8") as f:
            json.dump(tok_data, f)

        # 4. Run bundle creation
        bundle_out = os.path.join(self.tmpdir, "output_bundle.exe")
        res_path = create_bundle(
            model_path=fake_model,
            output_path=bundle_out,
            tokenizer_path=fake_tok_path,
            engine_path=fake_engine,
        )
        self.assertTrue(os.path.samefile(res_path, bundle_out))
        self.assertTrue(os.path.isfile(bundle_out))

        # 5. Inspect the generated bundle
        info = inspect_bundle(bundle_out)
        self.assertIsNotNone(info)
        self.assertEqual(info["engine_size"], real_engine_sz)
        self.assertEqual(info["model_size"], model_size)

        # 6. Strict 64 KB alignment validation
        self.assertEqual(
            info["vocab_offset"] % NATIVA_BUNDLE_ALIGN,
            0,
            f"vocab_offset ({info['vocab_offset']}) is not aligned to 64 KB",
        )
        self.assertEqual(
            info["model_offset"] % NATIVA_BUNDLE_ALIGN,
            0,
            f"model_offset ({info['model_offset']}) is not aligned to 64 KB",
        )
        self.assertGreaterEqual(info["vocab_offset"], real_engine_sz)
        self.assertGreaterEqual(info["model_offset"], info["vocab_offset"] + info["vocab_size"])

        # 7. Exact total file size
        expected_total = info["model_offset"] + info["model_size"] + TRAILER_SIZE
        self.assertEqual(os.path.getsize(bundle_out), expected_total)


class TestNativaBundleE2E(unittest.TestCase):
    """End-to-end test: builds a real executable and runs zero-GPU inference."""

    def test_real_bundle_build_and_inference(self):
        """Package the real qwen3_0.6b model with nativa_engine and run a self-contained benchmark."""
        is_win = sys.platform.startswith("win")
        engine_exe = "nativa_engine.exe" if is_win else "nativa_engine"
        real_engine = os.path.join(ROOT_DIR, "tools", engine_exe)
        real_model = os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4")
        real_tok = os.path.join(ROOT_DIR, "data", "qwen3_0.6b.natq4.tokenizer.json")

        if not (os.path.isfile(real_engine) and os.path.isfile(real_model) and os.path.isfile(real_tok)):
            self.skipTest("Real artifacts (nativa_engine, .natq4 model, or tokenizer) not found.")

        kwargs = {"ignore_cleanup_errors": True} if sys.version_info >= (3, 10) else {}
        with tempfile.TemporaryDirectory(**kwargs) as tmpdir:
            bundle_exe = os.path.join(tmpdir, "qwen3_bundle.exe" if is_win else "qwen3_bundle")
            try:
                # 1. Create the bundle
                create_bundle(
                    model_path=real_model,
                    output_path=bundle_exe,
                    tokenizer_path=real_tok,
                    engine_path=real_engine,
                )
                self.assertTrue(os.path.isfile(bundle_exe))

                # 2. Inspection
                meta = inspect_bundle(bundle_exe)
                self.assertIsNotNone(meta)
                self.assertEqual(meta["version"], 1)

                # 3. Self-contained execution with --bench 2
                cmd = [bundle_exe, "--bench", "2"]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=90)

                self.assertEqual(
                    proc.returncode,
                    0,
                    f"Bundle executable failed with code {proc.returncode}:\n{proc.stderr}\n{proc.stdout}",
                )

                # Check if the engine detected it is a bundle and completed prefill/generation
                self.assertTrue(
                    "[NATIVA BUNDLE] Executavel autocontido detectado!" in proc.stdout
                    or "[NATIVA BUNDLE] Self-contained executable detected!" in proc.stdout,
                    "Engine did not recognize the bundle trailer.",
                )
                self.assertTrue(
                    "[RESULTADO MOTOR UNIVERSAL .NATQ4]" in proc.stdout
                    or "[UNIVERSAL .NATQ4 ENGINE RESULT]" in proc.stdout,
                    "Engine did not complete inference successfully.",
                )
                self.assertIn(
                    "BENCH_HASH=",
                    proc.stdout,
                    "Engine did not generate benchmark hash.",
                )
            finally:
                # Ensure immediate cleanup of the >300 MB file with retries for WinError 32
                safe_unlink(bundle_exe)


if __name__ == "__main__":
    unittest.main(verbosity=2)
