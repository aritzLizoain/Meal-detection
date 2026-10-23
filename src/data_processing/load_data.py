from pathlib import Path
import pandas as pd


def load_data(eventlogs_dir, baselines_file):
    """
    Load all preprocessed eventlogs and baseline characteristics.

    Parameters
    ----------
    eventlogs_dir : str or Path
        Directory containing one eventlog CSV per participant.
    baselines_file : str or Path
        Path to the baseline characteristics CSV.

    Returns
    -------
    eventlogs : pd.DataFrame
        Combined eventlogs from all participants.
    baselines : pd.DataFrame
        Baseline characteristics.
    """

    eventlogs_dir = Path(eventlogs_dir)

    # Find all participant eventlogs
    files = sorted(eventlogs_dir.glob("*_eventlog.csv"))

    if not files:
        raise FileNotFoundError(
            f"No eventlog files found in {eventlogs_dir}"
        )

    # Load and combine eventlogs
    eventlogs = pd.concat(
        [pd.read_csv(file) for file in files],
        ignore_index=True
    )

    # Load baseline characteristics
    baselines = pd.read_csv(baselines_file)

    return eventlogs, baselines
