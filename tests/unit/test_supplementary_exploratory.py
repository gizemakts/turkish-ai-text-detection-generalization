from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SPEC=importlib.util.spec_from_file_location("supplementary_exploratory",ROOT/"scripts/supplementary_exploratory_validation.py")
MOD=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


def test_cluster_draw_keeps_members_together_and_is_replayable():
    units=np.array(["pair_a","pair_a","pair_b","pair_c","pair_c"])
    _,row_index,draws=MOD.cluster_draws(units,100,17)
    weights=draws[:,row_index]
    assert np.array_equal(weights[:,0],weights[:,1])
    assert np.array_equal(weights[:,3],weights[:,4])
    assert np.array_equal(draws,MOD.cluster_draws(units,100,17)[2])


def test_explicit_project_root_rejects_incomplete_directory(tmp_path):
    try:
        MOD.require_project(tmp_path)
    except RuntimeError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("incomplete root was accepted")
