import pytest
from ...conftest import mod_frontend


@pytest.fixture(scope="session")
def frontend():
    if mod_frontend["mxnet"]:
        return mod_frontend["mxnet"]
    return "mxnet"
