import pandas as pd


def get_baselines(baselines, subset, output_dir):
    """
    Calculate baseline and dataset characteristics
    for participants included in the analysis.
    """

    # Get IDs of included participants
    included_ids = subset["sid"].unique()

    # Filter baseline characteristics
    included_baselines = baselines[baselines["record_id"].isin(included_ids)].copy()

    # Calculate age statistics
    age = pd.to_numeric(included_baselines["age"], errors="coerce")

    # Get reported carbohydrate entries
    carbs = subset[subset["type"] == "carbs"].copy()

    # Calculate number of participant-days
    days = subset[["sid", "timestamp"]].copy()
    days["date"] = pd.to_datetime(days["timestamp"]).dt.date
    days = days[["sid", "date"]].drop_duplicates()

    # Calculate carbohydrate entries per participant-day
    carbs["date"] = pd.to_datetime(carbs["timestamp"]).dt.date

    carbs_per_day = (
        carbs.groupby(["sid", "date"])
        .size()
        .reindex(pd.MultiIndex.from_frame(days), fill_value=0)
    )

    # Calculate total reported carbohydrates per participant-day (g/day)
    carbs_grams_per_day = (
        carbs.groupby(["sid", "date"])["value"]
        .sum()
        .reindex(pd.MultiIndex.from_frame(days), fill_value=0)
    )

    # Calculate dataset characteristics
    baselines_table = pd.DataFrame(
        {
            "Characteristic": [
                "Participants",
                "Participant-days",
                "Age (years)",
                "Reported carbohydrate entries",
                "Carbohydrate entries per day",
                "Carbohydrate amount per entry (g)",
                "Reported carbohydrates per day (g)",
            ],
            "n": [
                len(included_ids),
                len(days),
                age.count(),
                len(carbs),
                len(carbs_per_day),
                len(carbs),
                len(carbs_grams_per_day),
            ],
            "mean": [
                None,
                None,
                round(age.mean(), 1),
                None,
                round(carbs_per_day.mean(), 1),
                round(carbs["value"].mean(), 1),
                round(carbs_grams_per_day.mean(), 1),
            ],
            "sd": [
                None,
                None,
                round(age.std(), 1),
                None,
                round(carbs_per_day.std(), 1),
                round(carbs["value"].std(), 1),
                round(carbs_grams_per_day.std(), 1),
            ],
            "min": [
                None,
                None,
                age.min(),
                None,
                carbs_per_day.min(),
                carbs["value"].min(),
                carbs_grams_per_day.min(),
            ],
            "max": [
                None,
                None,
                age.max(),
                None,
                carbs_per_day.max(),
                carbs["value"].max(),
                carbs_grams_per_day.max(),
            ],
        }
    )

    # Save table
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines_table.to_csv(output_dir / "baseline_characteristics.csv", index=False)
