import os

import pytest
from numpy._core._multiarray_umath import __cpu_features__

pytestmark = pytest.mark.skipif(
    os.environ.get("TWL_GOLDEN_STRICT") != "1",
    reason="CPU features are pinned only on the strict golden platform (D-023)",
)


def test_numpy_dispatches_x86_64_v3_without_avx512() -> None:
    assert __cpu_features__.get("X86_V3"), (
        "NumPy reports X86_V3 off; strict goldens need NPY_ENABLE_CPU_FEATURES=X86_V3 (D-023)"
    )
    assert not __cpu_features__.get("X86_V4"), (
        "NumPy reports X86_V4 (AVX-512) on; check NPY_ENABLE_CPU_FEATURES and the group names "
        "after any NumPy upgrade (D-023)"
    )
