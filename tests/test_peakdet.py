from ebfloeseg.peakdet import peakdet
import pytest


def test_peakdet():
    v = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
    delta = 1
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    peaks, valleys = peakdet(v, delta, x)
    peaks = [tuple(x) for x in peaks]
    valleys = [tuple(x) for x in valleys]
    assert peaks == [(3, 2), (5, 3), (7, 4), (9, 5)]
    assert valleys == [(4, 0), (6, 0), (8, 0)]

    v = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
    delta = 2
    peaks, valleys = peakdet(v, delta)
    assert [tuple(x) for x in peaks] == [(5, 3), (7, 4), (9, 5)]
    assert [tuple(x) for x in valleys] == [(6, 0), (8, 0)]

    # test with v and x of different lengths
    v = [0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0]
    x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    with pytest.raises(ValueError):
        peakdet(v, delta, x)

    # test with non-scalar delta
    delta = [1, 2]
    with pytest.raises(ValueError):
        peakdet(v, delta)

    # test with negative delta
    delta = -1
    with pytest.raises(ValueError):
        peakdet(v, delta)
    