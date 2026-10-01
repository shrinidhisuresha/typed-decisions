import pytest

torch = pytest.importorskip("torch")
from typed_decisions.backend_impls.transformers import best_device


class FakeBackends:
    def __init__(self, mps: bool):
        self.mps = type("mps", (), {"is_available": staticmethod(lambda: mps)})


def patch_torch(monkeypatch, cuda: bool, mps: bool):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends, "mps", FakeBackends(mps).mps)


def test_prefers_cuda_when_available(monkeypatch):
    patch_torch(monkeypatch, cuda=True, mps=True)
    assert best_device() == "cuda"


def test_falls_back_to_mps_on_apple_silicon(monkeypatch):
    # Not cosmetic: on this machine CPU is ~50x slower than MPS for the same model.
    patch_torch(monkeypatch, cuda=False, mps=True)
    assert best_device() == "mps"


def test_falls_back_to_cpu_when_no_accelerator_exists(monkeypatch):
    patch_torch(monkeypatch, cuda=False, mps=False)
    assert best_device() == "cpu"


def test_an_explicit_device_is_never_overridden(monkeypatch):
    patch_torch(monkeypatch, cuda=True, mps=True)
    assert best_device("cpu") == "cpu"
