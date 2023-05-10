import contextlib

import pytest


def pytest_collection_modifyitems(config, items):
    has_cupy = False
    with contextlib.suppress(ImportError):
        has_cupy = True

    skip_cupy = pytest.mark.skip(reason="cupy is not installed")
    for item in items:
        if "gpu" in item.keywords and not has_cupy:
            item.add_marker(skip_cupy)
