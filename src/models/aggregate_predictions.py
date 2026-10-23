import numpy as np
import pandas as pd


def aggregate_predictions(
    predictions,
    threshold=0.5,
    smoothing_window_min=15,
    max_gap_min=15,
    max_duration_min=60,
):
    """
    Convert timestamp-level scores into predicted meal events.

    - Centered moving average of prediction scores
    - Maximum 15-minute gap between candidate timestamps
    - Maximum 30-minute episode duration
    - First timestamp with the highest smoothed score is the onset

    Missing scores and recording gaps are not bridged.
    """
    if smoothing_window_min % 5 != 0 or smoothing_window_min < 5:
        raise ValueError("Smoothing window must be a positive multiple of 5.")

    if max_gap_min < 0 or max_duration_min < 0:
        raise ValueError("Gap and duration must be nonnegative.")

    window_size = smoothing_window_min // 5
    events = []

    for sid, group in predictions.groupby("sid", sort=False):
        group = group.sort_values("timestamp")

        times = pd.to_datetime(group["timestamp"]).to_numpy()
        scores = group["meal_probability"].to_numpy(dtype=float)

        # Separate recording segments wherever timestamps are not
        # exactly 5 minutes apart.
        gaps = np.flatnonzero(
            np.diff(times) != np.timedelta64(5, "m")
        ) + 1

        boundaries = np.r_[0, gaps, len(times)]

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment_times = times[start:end]
            segment_scores = scores[start:end]

            if len(segment_scores) < window_size:
                continue

            # Centered moving average. Require all values to be valid.
            smoothed = (
                pd.Series(segment_scores)
                .rolling(
                    window=window_size,
                    center=True,
                    min_periods=window_size,
                )
                .mean()
                .to_numpy()
            )

            candidate_idx = np.flatnonzero(smoothed >= threshold)

            if len(candidate_idx) == 0:
                continue

            # Build episodes without creating intermediate DataFrames.
            episode_start = 0

            def save_episode(first, last):
                indices = candidate_idx[first:last]

                # np.argmax selects the FIRST maximum.
                peak_idx = indices[np.argmax(smoothed[indices])]

                events.append({
                    "sid": sid,
                    "timestamp": segment_times[peak_idx],
                    "peak_score": smoothed[peak_idx],
                    "episode_start": segment_times[indices[0]],
                    "episode_end": segment_times[indices[-1]],
                    "n_points": len(indices),
                })

            for i in range(1, len(candidate_idx)):
                previous = candidate_idx[i - 1]
                current = candidate_idx[i]

                gap_min = (current - previous) * 5
                duration_min = (
                    current - candidate_idx[episode_start]
                ) * 5

                if gap_min > max_gap_min:
                    save_episode(episode_start, i)
                    episode_start = i

            save_episode(episode_start, len(candidate_idx))

    return pd.DataFrame(
        events,
        columns=[
            "sid",
            "timestamp",
            "peak_score",
            "episode_start",
            "episode_end",
            "n_points",
        ],
    )