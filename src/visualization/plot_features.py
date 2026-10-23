import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_features(features, subset, sid, date, output_file=None):
    """
    Plot extracted features for one participant-day.

    Reported carbohydrate events are shown for visualization only.
    """

    date = pd.Timestamp(date).normalize()
    next_date = date + pd.Timedelta(days=1)

    # --------------------------------------------------
    # Select participant-day
    # --------------------------------------------------

    day = features.loc[
        (features["sid"] == sid)
        & (features["timestamp"] >= date)
        & (features["timestamp"] < next_date)
    ].sort_values("timestamp")

    if day.empty:
        raise ValueError(
            f"No feature data for sid={sid}, date={date.date()}"
        )

    meals = subset.loc[
        (subset["sid"] == sid)
        & (subset["type"] == "carbs")
        & (subset["timestamp"] >= date)
        & (subset["timestamp"] < next_date)
        & (subset["value"] > 0)
    ].sort_values("timestamp")

    # --------------------------------------------------
    # Create figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        6, 1,
        figsize=(16, 16),
        sharex=True,
        constrained_layout=True
    )

    fig.suptitle(
        f"Participant {sid} | {date.date()}",
        fontsize=14
    )

    # 1. Glucose
    axes[0].plot(
        day["timestamp"],
        day["glucose_t"],
        color="tab:blue",
        label="CGM"
    )
    axes[0].set_ylabel("Glucose\n(mg/dL)")

    # 2. Glucose rate of change
    axes[1].plot(
        day["timestamp"],
        day["glucose_roc_60"],
        color="tab:green"
    )
    axes[1].axhline(0, color="gray", linewidth=0.8)
    axes[1].set_ylabel("Glucose ROC\n(mg/dL/min)")

    # 3. Maximum future glucose
    axes[2].plot(
        day["timestamp"],
        day["glucose_max_value_60"],
        color="tab:purple"
    )
    axes[2].set_ylabel("Max glucose\nnext 60 min\n(mg/dL)")

    # 4. Glucose AUC
    axes[3].plot(
        day["timestamp"],
        day["glucose_auc_60"],
        color="tab:red"
    )
    axes[3].axhline(0, color="gray", linewidth=0.8)
    axes[3].set_ylabel("Glucose AUC\n(mg/dL × min)")

    # 5. Insulin features
    axes[4].plot(
        day["timestamp"],
        day["insulin_past_30"],
        label="Past 30 min",
        color="tab:orange"
    )
    axes[4].plot(
        day["timestamp"],
        day["insulin_future_60"],
        label="Future 60 min",
        color="tab:brown"
    )
    axes[4].set_ylabel("Insulin (U)")
    axes[4].legend(loc="upper right")

    # 6. Time-of-day encoding
    axes[5].plot(
        day["timestamp"],
        day["hour_sin"],
        label="Hour sine",
        color="tab:cyan"
    )
    axes[5].plot(
        day["timestamp"],
        day["hour_cos"],
        label="Hour cosine",
        color="tab:pink"
    )
    axes[5].set_ylabel("Time encoding")
    axes[5].set_ylim(-1.1, 1.1)
    axes[5].legend(loc="upper right")

    # --------------------------------------------------
    # Reported carbohydrate events
    # --------------------------------------------------

    for ax in axes:
        for meal in meals.itertuples(index=False):
            ax.axvline(
                meal.timestamp,
                color="darkorange",
                linestyle="--",
                linewidth=1,
                alpha=0.7
            )

        ax.grid(alpha=0.2)

    for meal in meals.itertuples(index=False):
        axes[0].annotate(
            f"{meal.value:g} g",
            xy=(meal.timestamp, 1),
            xycoords=("data", "axes fraction"),
            xytext=(3, -5),
            textcoords="offset points",
            rotation=90,
            va="top",
            fontsize=8,
            color="darkorange"
        )

    # --------------------------------------------------
    # X-axis: 24-hour clock, consistent across panels
    # --------------------------------------------------

    axes[-1].set_xlim(date, next_date)
    axes[-1].xaxis.set_major_locator(
        mdates.HourLocator(interval=2)
    )
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )
    axes[-1].set_xlabel("Time of day")

    # --------------------------------------------------
    # Save and display
    # --------------------------------------------------

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()
    
    return fig