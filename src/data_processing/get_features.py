import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "glucose_t",
    "glucose_roc_60",
    "glucose_max_value_60",
    "glucose_auc_60",
    "insulin_past_30",
    "insulin_future_60",
    "hour_sin",
    "hour_cos",
]


def get_features(aligned):
    """
    Extract retrospective meal-detection features from aligned signals.

    Input columns:
        sid, timestamp, glucose, insulin

    Output:
        Original columns + feature columns.

    Windows:
        - Glucose: current time through next 60 minutes
        - Past insulin: preceding 30 minutes
        - Future insulin: current time through next 60 minutes

    Feature windows may cross midnight when adjacent data exist.
    Missing CGM values are not interpolated.
    Missing time intervals are not bridged.
    """

    results = []

    for sid, participant in aligned.groupby("sid", sort=False):

        participant = participant.sort_values("timestamp").copy()

        # Check that timestamps lie on the 5-minute grid.
        if not (
            participant["timestamp"].dt.minute.mod(5).eq(0).all()
            and participant["timestamp"].dt.second.eq(0).all()
            and participant["timestamp"].dt.microsecond.eq(0).all()
        ):
            raise ValueError(
                f"Invalid 5-minute timestamps for sid={sid}"
            )

        if participant["timestamp"].duplicated().any():
            raise ValueError(
                f"Duplicate timestamps for sid={sid}"
            )

        # Reindex to a continuous grid so missing days become NaN.
        # This prevents positional shifts from bridging data gaps.
        original_times = participant["timestamp"]

        full_grid = pd.date_range(
            participant["timestamp"].min(),
            participant["timestamp"].max(),
            freq="5min"
        )

        participant = (
            participant.set_index("timestamp")
            .reindex(full_grid)
        )

        participant.index.name = "timestamp"

        glucose = participant["glucose"]
        insulin = participant["insulin"]

        # --------------------------------------------------
        # Glucose features: next 60 minutes
        # --------------------------------------------------

        participant["glucose_t"] = glucose

        glucose_future = glucose.shift(-12)

        participant["glucose_roc_60"] = (
            glucose_future - glucose
        ) / 60

        future_glucose = pd.concat(
            [glucose.shift(-i) for i in range(13)],
            axis=1
        )

        valid_glucose_window = future_glucose.notna().all(axis=1)

        participant["glucose_max_value_60"] = (
            future_glucose.max(axis=1)
            .where(valid_glucose_window)
        )

        glucose_values = future_glucose.to_numpy(dtype=float)
        baseline = glucose.to_numpy(dtype=float)[:, None]

        relative_glucose = glucose_values - baseline

        auc = np.trapezoid(
            relative_glucose,
            dx=5,
            axis=1
        )

        participant["glucose_auc_60"] = np.where(
            valid_glucose_window.to_numpy(),
            auc,
            np.nan
        )

        # --------------------------------------------------
        # Insulin features
        # --------------------------------------------------

        participant["insulin_past_30"] = (
            insulin.shift(1)
            .rolling(window=6, min_periods=6)
            .sum()
        )

        future_insulin = pd.concat(
            [insulin.shift(-i) for i in range(12)],
            axis=1
        )

        participant["insulin_future_60"] = future_insulin.sum(
            axis=1,
            min_count=12
        )

        # --------------------------------------------------
        # Time-of-day features
        # --------------------------------------------------

        hour = (
            participant.index.hour
            + participant.index.minute / 60
        )

        angle = 2 * np.pi * hour / 24

        participant["hour_sin"] = np.sin(angle)
        participant["hour_cos"] = np.cos(angle)

        # Keep only timestamps present in the original aligned data.
        participant = participant.loc[original_times]

        participant = participant.reset_index()
        participant["sid"] = sid

        results.append(participant)

    return pd.concat(results, ignore_index=True)