import numpy as np
import torch

from openpi.shared.wrench_adapter import PolyFitWrenchDLA
from openpi.shared.wrench_adapter import WrenchAdapterConfig
from openpi.shared.wrench_adapter import build_robust_affine_adapter
from openpi.shared.wrench_adapter import load_adapter
from openpi.shared.wrench_adapter import save_adapter


def test_robust_affine_maps_location_and_scale() -> None:
    source = np.arange(600, dtype=np.float32).reshape(100, 6)
    target = source * 2.0 + 5.0
    adapter = build_robust_affine_adapter(source, target)
    result = adapter.transform_numpy(source)
    np.testing.assert_allclose(result.mean(axis=0), target.mean(axis=0), rtol=0.05, atol=0.1)


def test_polyfit_dla_is_bounded_and_roundtrips(tmp_path) -> None:
    source = np.linspace(-1.0, 1.0, 600, dtype=np.float32).reshape(100, 6)
    target = source * 2.0 + 5.0
    adapter = PolyFitWrenchDLA(
        source.mean(axis=0),
        np.maximum(source.std(axis=0), 1e-3),
        target.mean(axis=0),
        np.maximum(target.std(axis=0), 1e-3),
        config=WrenchAdapterConfig(hidden_dim=8, depth=1, dropout=0.0, residual_limit=1.0),
    )
    assert torch.isfinite(adapter(torch.from_numpy(source))).all()
    output = tmp_path / "adapter.pt"
    save_adapter(adapter, output)
    loaded = load_adapter(output)
    np.testing.assert_allclose(loaded.transform_numpy(source), adapter.transform_numpy(source), atol=1e-6)
