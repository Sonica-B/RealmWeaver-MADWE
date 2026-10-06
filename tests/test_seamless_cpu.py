import torch

from realmweaver.assets.seamless import set_seamless


def test_set_seamless_flips_every_padded_conv():
    m = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 3, padding=1),
        torch.nn.Conv2d(4, 4, 1),
        torch.nn.Conv2d(4, 4, 3, padding=1),
    )
    assert set_seamless(m, True) == 2
    assert all(c.padding_mode == "circular" for c in m if isinstance(c, torch.nn.Conv2d) and c.padding[0])
    assert m[1].padding_mode == "zeros"
    assert set_seamless(m, False) == 2 and m[0].padding_mode == "zeros"


def test_circular_padding_wraps_the_image_edge():
    """With circular padding an edge pixel sees the opposite edge, so a 3x3 mean filter of a one-hot
    image spreads mass across the seam; with zeros padding it does not."""
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1.0 / 9.0)
    x = torch.zeros(1, 1, 8, 8)
    x[0, 0, 0, 0] = 9.0
    set_seamless(conv, True)
    wrapped = conv(x)
    assert torch.isclose(wrapped[0, 0, 7, 7], torch.tensor(1.0))
    set_seamless(conv, False)
    assert conv(x)[0, 0, 7, 7] == 0.0
