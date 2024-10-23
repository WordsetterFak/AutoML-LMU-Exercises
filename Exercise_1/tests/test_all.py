from pathlib import Path

import numpy as np
from pytest import approx

from src.main import (Friedman_test, McNemar_test, Nemenyi_test,
                      TwoMatchedSamplest_test, load_data_FTest,
                      load_data_MNTest, load_data_TMStTest)

HERE = Path(__file__).parent.resolve()


def test_mcnemar_test():
    labels, predictions_1, predictions_2 = load_data_MNTest()
    value = McNemar_test(labels, predictions_1, predictions_2)

    assert value == approx(4.614, abs=1e-3)


def test_friedman():
    errors = load_data_FTest()
    value, _ = Friedman_test(errors=errors)

    assert value == approx(18.1333, abs=1e-3)


def test_two_matched_samples_test():
    y1, y2 = load_data_TMStTest()
    value = TwoMatchedSamplest_test(y1, y2)

    assert value == approx(-8.9235, abs=1e-3)


def test_nemenyi_test():
    expected = np.array(
        [
            [0.0, 0.92376043, 3.92598183, 1.84752086, 0.80829038],
            [0.0, 0.0, 3.0022214, 0.92376043, -0.11547005],
            [0.0, 0.0, 0.0, -2.07846097, -3.11769145],
            [0.0, 0.0, 0.0, 0.0, -1.03923048],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    errors = load_data_FTest()
    _, fdata_stats = Friedman_test(errors)

    value = Nemenyi_test(fdata_stats)

    np.testing.assert_almost_equal(value, expected, decimal=3)


# Add any tests you like, make sure to prefix it with
# `test`, i.e. `def test_mytest()`
