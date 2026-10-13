import pytest

torch = pytest.importorskip("torch")

from refdiff.model import CandNet, dihedral


def test_forward_shape():
    m = CandNet(width=8).eval()
    out = m(torch.zeros(2, 3, 64, 64, dtype=torch.uint8), torch.zeros(2, 4))
    assert out.shape == (2, 7)


def test_dihedral_group_and_side_swap():
    x = torch.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)
    s = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 2)
    seen = set()
    for k in range(8):
        y, t = dihedral(x, s, k)
        seen.add(tuple(y[0, 0].flatten().tolist()))
        swapped = bool(k & 4) ^ bool((k & 3) % 2)
        assert t[0, 2].item() == (4.0 if swapped else 3.0)
    assert len(seen) == 8
