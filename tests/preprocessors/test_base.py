import pandas as pd
import pytest
from ml.preprocessors.base import BasePreprocessor


def test_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        BasePreprocessor()


def test_fit_transform_calls_fit_then_transform(sample_X):
    class DummyPreprocessor(BasePreprocessor):
        def fit(self, X):
            self.fitted = True
            return self

        def transform(self, X):
            assert hasattr(self, "fitted"), "transform called before fit"
            return X

    p = DummyPreprocessor()
    result = p.fit_transform(sample_X)
    assert p.fitted
    pd.testing.assert_frame_equal(result, sample_X)
