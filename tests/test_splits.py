import numpy as np
import pandas as pd
import pytest

from earnedpass import config as C
from earnedpass.eval import splits


def toy():
    rows = []
    for code in C.FOLDERS:
        for level in C.LEVELS:
            for rep in range(1, 11):
                rows.append(dict(sample_id=f"C{code}{level}S{rep:02d}", code=code, level=level, cell=f"C{code}{level}",
                                 merged_group=f"C{code}{level}-{rep // 3}"))
    return pd.DataFrame(rows)


@pytest.mark.parametrize("rung", C.RUNGS)
def test_no_id_or_group_on_both_sides(rung):
    s = toy()
    fl = splits.folds(s, rung)
    assert len(fl) == {"1_sample_id": 5, "2_twin_aware": 5, "3_leave_cell_out": 39, "4_leave_folder_out": 3}[rung]
    seen = np.concatenate([te for _, _, te in fl])
    assert sorted(seen) == list(range(len(s)))  # every sample is tested exactly once
    for _, tr, te in fl:
        assert not set(tr) & set(te)


def test_check_catches_a_leak():
    s = toy()
    with pytest.raises(AssertionError):
        splits.check(s, np.array([0, 1]), np.array([1, 2]), "1_sample_id")
    with pytest.raises(AssertionError):
        splits.check(s, np.array([0]), np.array([1]), "3_leave_cell_out")  # same cell


def test_folds_respect_a_row_subset():
    s = toy()
    dev = np.flatnonzero(~s.level.isin(C.LOCKED_LEVELS).to_numpy())
    fl = splits.folds(s, "3_leave_cell_out", dev)
    assert len(fl) == 30
    for _, tr, te in fl:
        assert set(tr) <= set(dev) and set(te) <= set(dev)
