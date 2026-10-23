import pandas as pd


def aggregate_meals(subset, max_gap_min=15, max_duration_min=30):
    """
    Aggregate closely spaced carbohydrate entries into meal events.

    Entries are merged when:
    - They belong to the same participant.
    - The gap between consecutive entries is <= max_gap_min.
    - The time since the first entry of the meal is <= max_duration_min.

    Aggregation is continuous across midnight.

    The first carbohydrate entry defines the meal onset.
    """

    # Select positive carbohydrate entries
    carbs = subset[
        (subset["type"] == "carbs") &
        (subset["value"] > 0)
    ][["sid", "timestamp", "value"]].copy()

    carbs["timestamp"] = pd.to_datetime(carbs["timestamp"])
    carbs = carbs.sort_values(["sid", "timestamp"])

    meals = []

    # Process each participant independently (not each day)
    for sid, group in carbs.groupby("sid"):
        meal_start = None
        previous_time = None
        total_carbs = 0.0
        n_entries = 0

        for entry in group.itertuples(index=False):
            timestamp = entry.timestamp
            amount = entry.value

            if meal_start is None:
                # Start the first meal
                meal_start = timestamp
                total_carbs = amount
                n_entries = 1

            else:
                gap_min = (timestamp - previous_time).total_seconds() / 60
                duration_min = (timestamp - meal_start).total_seconds() / 60

                if gap_min <= max_gap_min and duration_min <= max_duration_min:
                    # Add entry to the current meal
                    total_carbs += amount
                    n_entries += 1

                else:
                    # Save the completed meal
                    meals.append({
                        "sid": sid,
                        "timestamp": meal_start,
                        "carbs": total_carbs,
                        "n_entries": n_entries,
                    })

                    # Start a new meal
                    meal_start = timestamp
                    total_carbs = amount
                    n_entries = 1

            previous_time = timestamp

        # Save the participant's final meal
        if meal_start is not None:
            meals.append({
                "sid": sid,
                "timestamp": meal_start,
                "carbs": total_carbs,
                "n_entries": n_entries,
            })

    return pd.DataFrame(
        meals,
        columns=["sid", "timestamp", "carbs", "n_entries"]
    )
