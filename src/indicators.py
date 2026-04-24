import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def roc(series: pd.Series, period: int) -> pd.Series:
    return (series / series.shift(period)) - 1.0


def consecutive_days_true(series: pd.Series) -> pd.Series:
    """Count rolling consecutive True values; resets to 0 on False."""
    result = []
    count = 0
    for val in series:
        if val:
            count += 1
        else:
            count = 0
        result.append(count)
    return pd.Series(result, index=series.index, dtype=int)
