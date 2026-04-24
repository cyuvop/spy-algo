import numpy as np
import pandas as pd
import pytest

from src.indicators import sma, roc, consecutive_days_true


class TestSma:
    def test_basic_three_period(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_period_one_equals_input(self):
        s = pd.Series([10.0, 20.0, 30.0])
        result = sma(s, 1)
        pd.testing.assert_series_equal(result, s.astype(float), check_names=False)

    def test_period_equals_length_gives_single_value(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = sma(s, 3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(2.0)

    def test_empty_series_returns_empty(self):
        s = pd.Series([], dtype=float)
        result = sma(s, 5)
        assert len(result) == 0

    def test_single_element(self):
        s = pd.Series([42.0])
        result = sma(s, 1)
        assert result.iloc[0] == pytest.approx(42.0)

    def test_single_element_period_two_is_nan(self):
        s = pd.Series([42.0])
        result = sma(s, 2)
        assert pd.isna(result.iloc[0])

    def test_all_nan_series(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        result = sma(s, 2)
        assert result.isna().all()

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx)
        result = sma(s, 3)
        assert list(result.index) == list(idx)


class TestRoc:
    def test_one_period_roc(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = roc(s, 1)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == pytest.approx(0.10)
        assert result.iloc[2] == pytest.approx(0.10)

    def test_two_period_roc(self):
        s = pd.Series([100.0, 110.0, 121.0])
        result = roc(s, 2)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(0.21)

    def test_negative_roc(self):
        s = pd.Series([100.0, 90.0])
        result = roc(s, 1)
        assert result.iloc[1] == pytest.approx(-0.10)

    def test_zero_roc_flat_series(self):
        s = pd.Series([50.0, 50.0, 50.0])
        result = roc(s, 1)
        assert result.iloc[1] == pytest.approx(0.0)
        assert result.iloc[2] == pytest.approx(0.0)

    def test_empty_series_returns_empty(self):
        s = pd.Series([], dtype=float)
        result = roc(s, 3)
        assert len(result) == 0

    def test_single_element_with_period_one_is_nan(self):
        s = pd.Series([100.0])
        result = roc(s, 1)
        assert pd.isna(result.iloc[0])

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        result = roc(s, 2)
        assert list(result.index) == list(idx)


class TestConsecutiveDaysTrue:
    def test_basic_sequence(self):
        s = pd.Series([True, True, False, True, True, True])
        result = consecutive_days_true(s)
        assert result.tolist() == [1, 2, 0, 1, 2, 3]

    def test_all_true(self):
        s = pd.Series([True, True, True, True])
        result = consecutive_days_true(s)
        assert result.tolist() == [1, 2, 3, 4]

    def test_all_false(self):
        s = pd.Series([False, False, False])
        result = consecutive_days_true(s)
        assert result.tolist() == [0, 0, 0]

    def test_alternating(self):
        s = pd.Series([True, False, True, False, True])
        result = consecutive_days_true(s)
        assert result.tolist() == [1, 0, 1, 0, 1]

    def test_single_true(self):
        s = pd.Series([True])
        result = consecutive_days_true(s)
        assert result.tolist() == [1]

    def test_single_false(self):
        s = pd.Series([False])
        result = consecutive_days_true(s)
        assert result.tolist() == [0]

    def test_empty_series(self):
        s = pd.Series([], dtype=bool)
        result = consecutive_days_true(s)
        assert len(result) == 0

    def test_reset_to_zero_on_false_then_rebuilds(self):
        s = pd.Series([True, True, True, False, True, True])
        result = consecutive_days_true(s)
        assert result.tolist() == [1, 2, 3, 0, 1, 2]

    def test_preserves_index(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="D")
        s = pd.Series([True, True, False, True], index=idx)
        result = consecutive_days_true(s)
        assert list(result.index) == list(idx)
