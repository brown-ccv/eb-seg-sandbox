import numpy as np
from numpy.typing import ArrayLike


def peakdet(v: ArrayLike, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Detects peaks and valleys in a given input vector.

    Parameters:
    v (array-like): The input vector.
    delta (float): The minimum difference between a peak (or valley) and its surrounding points.
    x (array-like, optional): The x-coordinates corresponding to the input vector. If not provided, the indices of the input vector will be used.

    Returns:
    tuple: A tuple containing two arrays. The first array contains the detected peaks, and the second array contains the detected valleys.

    Raises:
    ValueError: If the lengths of the input vectors `v` and `x` are not the same.
    ValueError: If the input argument `delta` is not a scalar.
    ValueError: If the input argument `delta` is not positive.

    """

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        raise ValueError("Input vectors v and x must have the same length")

    if not np.isscalar(delta):
        raise ValueError("Input argument delta must be a scalar")

    if delta <= 0:
        raise ValueError("Input argument delta must be positive")

    mn, mx = np.Inf, -np.Inf

    lookformax = True
    maxtab = []
    mintab = []

    for i, this in enumerate(v):
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
