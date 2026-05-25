"""TimeSeriesSplit with embargo — exact pattern from the PDF guide §5.

Why embargo: the engineered features include lag/rolling windows. The last
few training rows can therefore contain information that is statistically
indistinguishable from the first few validation rows, even though the
period_months are strictly sequential. Inserting a small `embargo` gap
(in rows) between train_end and valid_start cuts that bleed.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def ts_folds(
    n: int,
    n_splits: int = 5,
    embargo: int = 1,
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, valid_idx) tuples.

    Args:
        n: total number of rows in the feature matrix
        n_splits: how many CV folds (sklearn TimeSeriesSplit semantics)
        embargo: number of rows to drop from the END of train, immediately
                 before valid_start. embargo=0 disables (NOT recommended
                 when features contain rolling windows).
    """
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0 (got {embargo})")
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2 (got {n_splits})")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    for tr_idx, va_idx in tscv.split(np.arange(n)):
        # Cut the last `embargo` train rows so the rolling-window tail of
        # the train fold cannot bleed into the valid fold's first rows.
        if embargo > 0:
            cutoff = va_idx[0] - embargo
            tr_idx = tr_idx[tr_idx < cutoff]
        # Skip degenerate folds (could happen if embargo >= len(train) for
        # very small first folds).
        if len(tr_idx) == 0:
            continue
        yield tr_idx, va_idx
