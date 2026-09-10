"""Run the tests the way the Space runs: this directory on the path and as the working
directory, because app.py refers to `samples/...` relatively, as a Space should."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


@pytest.fixture(autouse=True, scope="session")
def _cwd_is_the_space():
    prev = os.getcwd()
    os.chdir(HERE)
    yield
    os.chdir(prev)
