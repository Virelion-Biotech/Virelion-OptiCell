from pathlib import Path

import numpy as np

from scripts.run_killer_workflow import default_phenotype_rules, load_gray


def test_killer_load_gray_returns_native_and_uint8_views(tmp_path: Path):
    import tifffile

    image = np.zeros((40, 40), dtype=np.uint16)
    image[10:20, 10:20] = 4095
    path = tmp_path / "native16.tiff"
    tifffile.imwrite(path, image)

    native_gray, gray8 = load_gray(path)

    assert native_gray.dtype == np.uint16
    assert native_gray.shape == image.shape
    assert float(native_gray.max()) == 4095.0
    assert gray8.dtype == np.uint8
    assert gray8.shape == image.shape


def test_default_phenotype_rules_are_not_bit_depth_dependent():
    labels = {rule.label for rule in default_phenotype_rules()}
    assert labels == {"area_ok", "roundish"}
