import pandas as pd
import matplotlib.pyplot as plt


def filter_IDs_days(
    subset,
    meals,
    min_meals_day=3,
    max_meals_day=8,
    min_days_id=14,
    consecutive=True,
    output_file=None,
):
    """
    Filter participant-days by number of aggregated meals >=5 g.

    If consecutive=True:
        Keep only consecutive stretches containing at least
        min_days_id valid days.

    If consecutive=False:
        Keep participants with at least min_days_id valid days
        in total, even if those days are not consecutive.

    Returns
    -------
    filtered_subset, filtered_meals
    """

    # Original participant-days, including days with zero meals
    days = (
        subset[["sid", "timestamp"]]
        .assign(date=lambda x: x["timestamp"].dt.normalize())
        [["sid", "date"]]
        .drop_duplicates()
    )

    # Count aggregated meals >=5 g per participant-day
    meal_counts = (
        meals.loc[meals["carbs"] >= 5, ["sid", "timestamp"]]
        .assign(date=lambda x: x["timestamp"].dt.normalize())
        .groupby(["sid", "date"])
        .size()
        .rename("n_meals")
        .reset_index()
    )

    days = days.merge(
        meal_counts,
        on=["sid", "date"],
        how="left",
    )

    days["n_meals"] = days["n_meals"].fillna(0).astype(int)

    # Original participant-level statistics
    original = (
        days.groupby("sid")
        .agg(
            original_days=("date", "size"),
            days_with_meals=("n_meals", lambda x: (x > 0).sum()),
            mean_meals=("n_meals", lambda x: x[x > 0].mean()),
        )
    )

    # Filter days by meal count
    kept_days = days.loc[
        days["n_meals"].between(min_meals_day, max_meals_day)
    ].copy()

    # Filter by minimum number of days
    if consecutive:

        # Sort valid days chronologically
        kept_days = kept_days.sort_values(["sid", "date"]).copy()

        # Start a new streak whenever valid days are not consecutive
        kept_days["streak"] = (
            kept_days.groupby("sid")["date"]
            .diff()
            .ne(pd.Timedelta(days=1))
            .groupby(kept_days["sid"])
            .cumsum()
        )

        # Calculate length of each consecutive streak
        kept_days["streak_length"] = (
            kept_days.groupby(["sid", "streak"])["date"]
            .transform("size")
        )

        # Retain only sufficiently long consecutive stretches
        kept_days = kept_days.loc[
            kept_days["streak_length"] >= min_days_id
        ].copy()

    else:

        # Count all valid days per participant
        retained_days_per_id = kept_days.groupby("sid").size()

        # Retain participants with enough valid days
        retained_ids = retained_days_per_id[
            retained_days_per_id >= min_days_id
        ].index

        kept_days = kept_days.loc[
            kept_days["sid"].isin(retained_ids)
        ].copy()

    retained_ids = kept_days["sid"].unique()

    # Participant-level statistics after filtering
    retained = (
        kept_days.groupby("sid")
        .agg(retained_days=("date", "size"))
    )

    summary = original.join(retained, how="left")

    summary["retained_days"] = (
        summary["retained_days"].fillna(0).astype(int)
    )

    # Filter original eventlogs
    subset_with_date = subset.assign(
        date=subset["timestamp"].dt.normalize()
    )

    filtered_subset = (
        subset_with_date.merge(
            kept_days[["sid", "date"]],
            on=["sid", "date"],
            how="inner",
        )
        .drop(columns="date")
    )

    # Filter aggregated meals using the same participant-days
    meals_with_date = meals.assign(
        date=meals["timestamp"].dt.normalize()
    )

    filtered_meals = (
        meals_with_date.merge(
            kept_days[["sid", "date"]],
            on=["sid", "date"],
            how="inner",
        )
        .drop(columns="date")
    )

    # Plot original versus retained participant-days
    if output_file is not None:

        fig, ax = plt.subplots(figsize=(12, 7))

        size_scale = 2.0

        # Original participants
        ax.scatter(
            summary["days_with_meals"],
            summary["mean_meals"],
            s=summary["original_days"] * size_scale,
            color="gray",
            edgecolor="black",
            alpha=0.65,
            label="Original",
        )

        # Retained participants
        kept = summary["retained_days"] > 0

        ax.scatter(
            summary.loc[kept, "days_with_meals"],
            summary.loc[kept, "mean_meals"],
            s=summary.loc[kept, "retained_days"] * size_scale,
            color="red",
            alpha=0.9,
            label="Retained",
        )

        ax.set(
            xlabel="Original days with ≥1 reported meal",
            ylabel="Original average meals per day (days with meals)",
            title="Meal reporting: original vs retained participant-days",
        )

        ax.legend()
        ax.grid(alpha=0.2)

        fig.tight_layout()
        fig.savefig(output_file, dpi=300)
        plt.close(fig)

    # Print filtering summary
    print("\n--- Meal-frequency filtering ---")
    print(f"Consecutive days required: {consecutive}")
    print(f"Meal range per day: {min_meals_day}–{max_meals_day}")
    print(f"Minimum days: {min_days_id}")
    print(f"Original participants: {len(original)}")
    print(f"Retained participants: {len(retained_ids)}")
    print(f"Original participant-days: {len(days):,}")
    print(f"Retained participant-days: {len(kept_days):,}")
    print(f"Original aggregated meals: {len(meals):,}")
    print(f"Retained aggregated meals: {len(filtered_meals):,}")

    return filtered_subset, filtered_meals