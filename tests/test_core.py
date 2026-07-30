import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "xafs_wavelet_studio.py"
SPEC = importlib.util.spec_from_file_location("xafs_wavelet_studio", MODULE_PATH)
studio = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = studio
SPEC.loader.exec_module(studio)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.k, self.chi = studio.generate_demo_data()

    def test_morlet_shape_and_expected_shell_region(self):
        settings = studio.TransformSettings(k_points=64, r_points=48)
        result = studio.calculate_transform(self.k, self.chi, settings)
        self.assertEqual(result.coefficients.shape, (48, 64))
        self.assertTrue(np.isfinite(result.coefficients).all())
        peak_r = result.r[np.argmax(np.sum(result.magnitude, axis=1))]
        self.assertGreater(peak_r, 1.5)
        self.assertLess(peak_r, 3.6)

    def test_cauchy_is_finite_and_nonzero(self):
        settings = studio.TransformSettings(
            wavelet="Cauchy (EXAFS)", k_points=60, r_points=44,
            cauchy_nfft=1024,
        )
        result = studio.calculate_transform(self.k, self.chi, settings)
        self.assertEqual(result.coefficients.shape, (44, 60))
        self.assertTrue(np.isfinite(result.coefficients).all())
        self.assertGreater(float(result.magnitude.max()), 0.0)

    def test_auto_dk_handles_irregular_samples(self):
        keep = np.ones(self.k.size, dtype=bool)
        keep[20::17] = False
        settings = studio.TransformSettings(k_points=48, r_points=36)
        result = studio.calculate_transform(self.k[keep], self.chi[keep], settings)
        self.assertTrue(np.isfinite(result.coefficients).all())
        self.assertAlmostEqual(result.integration_dk, 0.04, places=6)

    def test_npz_export_includes_metadata_and_complex_values(self):
        settings = studio.TransformSettings(k_points=44, r_points=32)
        result = studio.calculate_transform(self.k, self.chi, settings, "unit test")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.npz"
            studio.export_transform(path, result)
            with np.load(path) as exported:
                self.assertEqual(exported["coefficients"].shape, (32, 44))
                metadata = json.loads(str(exported["metadata"]))
                self.assertEqual(metadata["author"], "Dr. Esmael Balaghi")
                self.assertEqual(metadata["source"], "unit test")


if __name__ == "__main__":
    unittest.main()

