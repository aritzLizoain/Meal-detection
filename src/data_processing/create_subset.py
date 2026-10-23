import pandas as pd


def create_subset(eventlogs, runnable_cases_file):
    """
    Select eventlogs for participant-date combinations
    for which the SI model can run.

    Parameters
    ----------
    eventlogs : pd.DataFrame
        All preprocessed eventlogs.
    runnable_cases_file : str or Path
        Path to runnable_cases.csv.

    Returns
    -------
    subset : pd.DataFrame
        Eventlogs restricted to runnable participant-days.
    """

    # Load runnable cases
    runnable_cases = pd.read_csv(runnable_cases_file)

    # Convert timestamps to datetime
    eventlogs = eventlogs.copy()
    eventlogs["timestamp"] = pd.to_datetime(eventlogs["timestamp"])

    # Extract date from eventlogs
    eventlogs["date"] = eventlogs["timestamp"].dt.normalize()

    # Convert runnable dates to datetime
    runnable_cases["date"] = pd.to_datetime(
        runnable_cases["date"]
    ).dt.normalize()

    # Filter eventlogs
    subset = eventlogs.merge(
        runnable_cases,
        on=["sid", "date"],
        how="inner"
    )

    # Remove temporary date column
    subset = subset.drop(columns="date")
    
    # Remove carbohydrate entries with zero or negative values
    subset = subset[
        ~((subset["type"] == "carbs") & (subset["value"] <= 0))
    ].copy()
    
    # Exclude participant 505 due to nonstandard carbohydrate reporting with DIY Loop.
    # The extended-meal feature splits a single meal into multiple small carbohydrate
    # entries spread over time, resulting in >30 entries/day on average.
    # This makes it difficult to identify individual meal events reliably.
    subset = subset[subset["sid"] != 505].copy()

    return subset
