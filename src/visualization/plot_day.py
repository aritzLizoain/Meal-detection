import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_day(
    data,
    sid,
    date,
    output_file=None,
    predicted_events=None,
    meals=None,
    matches=None,
    method=None,
):
    """
    Plot one participant-day from raw eventlogs or aligned signals.

    Raw columns:
        sid, timestamp, type, value, unit, detail

    Aligned columns:
        sid, timestamp, glucose, insulin
        insulin is delivered units per 5-minute interval.
    """

    day_start = pd.Timestamp(date)
    day_end = day_start + pd.Timedelta(days=1)

    day = data.loc[
        (data["sid"] == sid)
        & (data["timestamp"] >= day_start)
        & (data["timestamp"] < day_end)
    ].sort_values("timestamp")

    if day.empty:
        raise ValueError(f"No data for sid={sid}, date={date}")

    is_raw = {"type", "value"}.issubset(day.columns)

    fig, axes = plt.subplots(
        3 if is_raw else 2,
        1,
        figsize=(14, 8 if is_raw else 6),
        sharex=True,
        constrained_layout=True
    )

    if is_raw:

        bg = day[day["type"] == "bg"]
        basal = day[day["type"] == "basal"]
        bolus = day[day["type"] == "bolus"]
        carbs = day[day["type"] == "carbs"]

        # CGM
        axes[0].plot(
            bg["timestamp"],
            bg["value"],
            marker=".",
            markersize=4,
            linewidth=1,
            label="CGM"
        )
        axes[0].set_ylabel("Glucose (mg/dL)")

        # Basal rate
        axes[1].step(
            basal["timestamp"],
            basal["value"],
            where="post",
            label="Basal"
        )
        axes[1].set_ylabel("Basal (U/min)")

        # Bolus impulses
        axes[2].vlines(
            bolus["timestamp"],
            0,
            bolus["value"],
            label="Bolus"
        )
        axes[2].set_ylabel("Bolus (U)")

        # Reported carbohydrate events
        for i, (_, meal) in enumerate(carbs.iterrows()):
            axes[0].axvline(
                meal["timestamp"],
                color="tab:orange",
                linestyle="--",
                alpha=0.7,
                label="Reported carbs" if i == 0 else None
            )
        
            axes[0].annotate(
                f'{meal["value"]:g} g',
                xy=(meal["timestamp"], 1),
                xycoords=("data", "axes fraction"),
                xytext=(3, -5),
                textcoords="offset points",
                rotation=90,
                ha="left",
                va="top",
                fontsize=8,
                color="tab:orange"
            )

        title = f"Raw eventlog | Participant {sid} | {day_start.date()}"

    else:

        required = {"glucose", "insulin"}
        if not required.issubset(day.columns):
            raise ValueError(
                f"Aligned data must contain {required}"
            )

        # Aligned CGM
        axes[0].plot(
            day["timestamp"],
            day["glucose"],
            marker=".",
            markersize=4,
            linewidth=1,
            label="CGM"
        )
        axes[0].set_ylabel("Glucose (mg/dL)")

        # Total delivered insulin
        axes[1].bar(
            day["timestamp"],
            day["insulin"],
            width=pd.Timedelta(minutes=4),
            label="Total insulin"
        )
        axes[1].set_ylabel("Insulin (U / 5 min)")

        title = f"Aligned signals | Participant {sid} | {day_start.date()}"

        # Reported meals and predicted meal events (optional)
        if meals is not None:
    
            day_meals = meals.loc[
                (meals["sid"] == sid)
                & (meals["timestamp"] >= day_start)
                & (meals["timestamp"] < day_end)
            ].sort_values("timestamp")
    
            for i, meal in enumerate(day_meals.itertuples()):
                axes[0].axvline(
                    meal.timestamp,
                    color="tab:orange",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.8,
                    label="Reported meal" if i == 0 else None,
                )
    
                axes[0].annotate(
                    f"{meal.carbs:g} g",
                    xy=(meal.timestamp, 1),
                    xycoords=("data", "axes fraction"),
                    xytext=(3, -5),
                    textcoords="offset points",
                    rotation=90,
                    ha="left",
                    va="top",
                    fontsize=8,
                    color="tab:orange",
                )
    
        if predicted_events is not None:
    
            day_predictions = predicted_events.loc[
                (predicted_events["sid"] == sid)
                & (predicted_events["timestamp"] >= day_start)
                & (predicted_events["timestamp"] < day_end)
            ].sort_values("timestamp")
    
            # Use the predicted-event timestamps from match_events()
            matched_times = set()
    
            if matches is not None and not matches.empty:
                matched_times = set(
                    matches.loc[
                        matches["sid"] == sid,
                        "predicted_timestamp",
                    ]
                )
    
            shown_labels = set()
    
            for event in day_predictions.itertuples():
    
                is_matched = event.timestamp in matched_times
    
                color = "tab:green" if is_matched else "tab:red"
                label = (
                    "Matched prediction"
                    if is_matched
                    else "Unmatched prediction"
                )
    
                axes[0].axvline(
                    event.timestamp,
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    alpha=0.85,
                    label=label if label not in shown_labels else None,
                )
    
                shown_labels.add(label)
    
        if method is not None:
            title += f" | {method}"
            
    # Shared formatting
    axes[0].set_title(title)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        ax.set_xlim(day_start, day_end)

    axes[-1].xaxis.set_major_locator(
        mdates.HourLocator(interval=2)
    )
    axes[-1].xaxis.set_major_formatter(
        mdates.DateFormatter("%H:%M")
    )
    axes[-1].set_xlabel("Time")

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

    return fig