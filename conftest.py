import pytest

from app import tools
from data.make_data import build


@pytest.fixture(scope="session", autouse=True)
def db(tmp_path_factory):
    path = build(tmp_path_factory.mktemp("db") / "inventory.db")
    tools.DB = path
    return path
