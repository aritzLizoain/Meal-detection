import numpy as np
import pandas as pd


def create_labels(
    features,
    meals,
    min_carbs=5,
    positive_window_min=10,
    ambiguous_window_min=60,
):
    """
    Create retrospective meal-onset labels.

    1   Within +/- 10 min of a reported meal >= 5 g
    NaN Within +/- 60 min of any reported carbohydrate event
    0   Otherwise (provisional negative)

    Positive labels take precedence over ambiguous labels.
    """
    if not 0 <= positive_window_min <= ambiguous_window_min:
        raise ValueError("Invalid labeling windows.")

    labeled = features.copy()
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"])
    labeled["label"] = 0.0

    meals = meals.copy()
    meals["timestamp"] = pd.to_datetime(meals["timestamp"])

    # Work with sorted timestamps for efficient window lookup.
    for sid, group in labeled.groupby("sid"):
        participant_meals = meals[meals["sid"] == sid]

        if participant_meals.empty:
            continue

        group = group.sort_values("timestamp")
        times = group["timestamp"].to_numpy()
        labels = np.zeros(len(group), dtype=float)

        def mark_windows(meal_times, window_min, value):
            window = np.timedelta64(window_min, "m")

            for meal_time in meal_times:
                start = np.searchsorted(times, meal_time - window)
                end = np.searchsorted(times, meal_time + window, side="right")
                labels[start:end] = value

        # First mark ambiguous windows around ALL carb events.
        mark_windows(
            participant_meals["timestamp"].to_numpy(),
            ambiguous_window_min,
            np.nan,
        )

        # Then mark positives around eligible meals.
        eligible = participant_meals[
            participant_meals["carbs"] >= min_carbs
        ]

        mark_windows(
            eligible["timestamp"].to_numpy(),
            positive_window_min,
            1.0,
        )

        labeled.loc[group.index, "label"] = labels

    return labeled