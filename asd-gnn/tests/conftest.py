from __future__ import annotations

import numpy as np
import pytest

from asdgnn.data.synthetic import make_cohort, make_conn

N_TOTAL = 60


@pytest.fixture(scope="session")
def cohort():
    ts, pheno = make_cohort(n_subjects=N_TOTAL, n_rois=12, n_sites=4, seed=0)
    pheno = pheno.loc[sorted(ts)]
    return [ts[s] for s in pheno.index], pheno


@pytest.fixture(scope="session")
def dummy_data(cohort):
    return cohort[0]


@pytest.fixture(scope="session")
def split_idx():
    rng = np.random.RandomState(0)
    perm = rng.permutation(N_TOTAL)
    return np.sort(perm[:45]), np.sort(perm[45:])


@pytest.fixture(scope="session")
def conn_and_y():
    return make_conn(n=24, n_rois=10, seed=1)


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: needs torch / torch-geometric")


@pytest.fixture(scope="session")
def torch_available():
    try:
        import torch_geometric  # noqa: F401
        return True
    except ImportError:
        return False
