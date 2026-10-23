import numpy as np
import pandas as pd


def align_signals(subset):
    """
    Align CGM and insulin delivery to a 5-minute grid.

    Output columns:
        sid, timestamp, glucose, insulin

    glucose: mg/dL
    insulin: total delivered U per 5-minute interval

    Assumptions:
        - Basal value: U/min
        - Basal detail: duration in minutes
        - Bolus value: delivered U
        - Missing CGM values are not interpolated
    """

    results = []
    interval = pd.Timedelta(minutes=5)

    for sid, participant in subset.groupby("sid", sort=False):

        participant = participant.sort_values("timestamp").copy()

        # --------------------------------------------------
        # Construct grid for observed participant-days
        # --------------------------------------------------

        dates = (
            participant["timestamp"]
            .dt.normalize()
            .drop_duplicates()
            .sort_values()
        )

        grid = pd.DatetimeIndex(
            np.concatenate([
                pd.date_range(
                    date,
                    date + pd.Timedelta(days=1),
                    freq="5min",
                    inclusive="left",
                ).values
                for date in dates
            ])
        )

        aligned = pd.DataFrame({
            "sid": sid,
            "timestamp": grid,
            "glucose": np.nan,
            "insulin": 0.0,
        })

        # --------------------------------------------------
        # CGM: nearest 5-minute grid point
        # --------------------------------------------------

        bg = participant.loc[
            participant["type"] == "bg",
            ["timestamp", "value"],
        ].copy()

        bg["grid_time"] = bg["timestamp"].dt.round("5min")

        glucose_map = bg.groupby("grid_time")["value"].mean()

        aligned["glucose"] = aligned["timestamp"].map(glucose_map)

        # --------------------------------------------------
        # Basal: exact interval overlap
        # --------------------------------------------------

        basal = participant.loc[
            participant["type"] == "basal",
            ["timestamp", "value", "detail"],
        ].copy()

        basal["rate"] = pd.to_numeric(
            basal["value"], errors="coerce"
        )

        basal["duration"] = pd.to_numeric(
            basal["detail"], errors="coerce"
        )

        if basal[["rate", "duration"]].isna().any().any():
            raise ValueError(f"Invalid basal data for sid={sid}")

        if (basal[["rate", "duration"]] < 0).any().any():
            raise ValueError(f"Negative basal data for sid={sid}")

        insulin = pd.Series(0.0, index=grid)

        # Basal: fast, exact interval-overlap allocation
        insulin = np.zeros(len(grid), dtype=float)

        # Explicit nanosecond resolution, independent of pandas defaults
        grid_ns = grid.to_numpy(dtype="datetime64[ns]").astype("int64")

        start_ns = (
            basal["timestamp"]
            .to_numpy(dtype="datetime64[ns]")
            .astype("int64")
        )

        duration_ns = (
            pd.to_timedelta(basal["duration"], unit="min")
            .to_numpy(dtype="timedelta64[ns]")
            .astype("int64")
        )

        end_ns = start_ns + duration_ns
        rates = basal["rate"].to_numpy(dtype=float)

        interval_ns = 5 * 60 * 1_000_000_000
        ns_per_minute = 60 * 1_000_000_000

        for start, end, rate in zip(start_ns, end_ns, rates):

            if end <= start or rate == 0:
                continue

            # Candidate bins whose intervals may overlap this event
            first = np.searchsorted(
                grid_ns, start - interval_ns + 1, side="left"
            )
            last = np.searchsorted(
                grid_ns, end, side="left"
            )

            if first >= last:
                continue

            indices = np.arange(first, last)

            bin_starts = grid_ns[indices]
            bin_ends = bin_starts + interval_ns

            overlap_ns = np.maximum(
                0,
                np.minimum(end, bin_ends)
                - np.maximum(start, bin_starts),
            )

            insulin[indices] += rate * overlap_ns / ns_per_minute

        # --------------------------------------------------
        # Bolus: assign impulses to containing 5-minute bin
        # --------------------------------------------------

        bolus = participant.loc[
            participant["type"] == "bolus",
            ["timestamp", "value"],
        ].copy()

        bolus["value"] = pd.to_numeric(
            bolus["value"], errors="coerce"
        )

        if bolus["value"].isna().any():
            raise ValueError(f"Invalid bolus data for sid={sid}")

        if (bolus["value"] < 0).any():
            raise ValueError(f"Negative bolus data for sid={sid}")

        bolus["grid_time"] = bolus["timestamp"].dt.floor("5min")
        bolus_per_bin = bolus.groupby("grid_time")["value"].sum()

        bolus_aligned = (
            pd.Series(grid)
            .map(bolus_per_bin)
            .fillna(0.0)
            .to_numpy()
        )

        insulin += bolus_aligned
        aligned["insulin"] = insulin

        results.append(aligned)

    return pd.concat(results, ignore_index=True)