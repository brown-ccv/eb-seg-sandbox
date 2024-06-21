import numpy as np
from numpy.typing import ArrayLike


def peakdet(v: ArrayLike, delta: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Detects peaks and valleys in a given input vector.

    Parameters:
    v (array-like): The input vector.
    delta (float): The minimum difference between a peak (or valley) and its surrounding points.

    Returns:
    tuple: A tuple containing two arrays. The first array contains the detected peaks, and the second array contains the detected valleys.

    Raises:
    ValueError: If the input argument `delta` is not a scalar.
    ValueError: If the input argument `delta` is not positive.

    """

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
            mxpos = i
        if this < mn:
            mn = this
            mnpos = i

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = i
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = i
                lookformax = True

    return np.array(maxtab), np.array(mintab)
