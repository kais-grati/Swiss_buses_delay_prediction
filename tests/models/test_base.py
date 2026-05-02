import pytest
from ml.models.base import BaseModel


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        BaseModel()
