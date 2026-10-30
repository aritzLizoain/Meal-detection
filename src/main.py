import config
import pandas as pd

from data_processing import (
    load_data,
    create_subset,
    get_baselines,
    get_features,
    align_signals,
    aggregate_meals,
    create_labels,
    filter_IDs_days,
)

from visualization import (
    plot_day,
    plot_features,
)

from evaluation import (
    participant_cv,
    match_events,
)

from models import (
    aggregate_predictions,
    rule_based_detector,
    logistic_regression_detector,
    xgboost_detector
)

def main():

    # Read preprocessed eventlogs and baseline data
    eventlogs, baselines = load_data(
        eventlogs_dir=config.EVENTLOGS_DIR,
        baselines_file=config.BASELINES_FILE
    )
    
    # Create common analysis subset
    subset = create_subset(
        eventlogs,
        config.RUNNABLE_CASES
    )
    
    # Combine carb entries into meal events
    meals = aggregate_meals(
        subset,
        max_gap_min= 15,
        max_duration_min=30
    )
    
    # Filter IDs and days based on aggregated meal events
    subset, meals = filter_IDs_days(
        subset,
        meals,
        min_meals_day=3,
        max_meals_day=6,
        min_days_id=14,
        consecutive=True,
        output_file=config.FIGURES_DIR / "meal_frequency_filtering.png",
    )
    
    # Baseline characteristics of retained participants
    get_baselines(
        baselines,
        subset,
        output_dir=config.TABLES_DIR
    )
    
    # Visualize example day
    plot_day(
        subset,
        sid=14,
        date="2023-09-28",
        output_file=config.FIGURES_DIR / "raw_day_14_2023-09-28.png"
    )
    
    # Align data to 5-min grid
    aligned = align_signals(subset)
    
    # Visualize aligned example day
    plot_day(
        aligned,
        sid=14,
        date="2023-09-28",
        output_file=config.FIGURES_DIR / "aligned_day_14_2023-09-28.png"
    )
    
    # Extract features
    features = get_features(aligned)
    
    # Plot features
    plot_features(
        features=features,
        subset=subset,
        sid=14,
        date="2023-09-28",
        output_file=config.FIGURES_DIR / "features_14_2023-09-28.png"
    )
    
    # Create training labels
    labeled = create_labels(
        features,
        meals,
        min_carbs=5,
        positive_window_min=10,
        ambiguous_window_min=60,
    )
    
    """
    Meal detection methods
    1) Rule-based detection
    2) Logistic regression
    3) XGBoost
    4) LSTM
    """

    predictions = {
        "rule_based": [],
        "logistic_regression": [],
        "xgboost": [],
        "lstm": [],
    }
        
    # Cross-validation: generate out-of-fold predictions   
    for fold, train, val in participant_cv(labeled, n_splits= 5):

        print(f"\n--- Fold {fold}/5 ---")
        print(
            f"Training: {train['sid'].nunique()} participants | "
            f"Validation: {val['sid'].nunique()} participants"
        )

        # Rule-based detector
        print("Running rule-based detector...")

        rule_pred = rule_based_detector(val, min_roc=0.5, min_auc=300)
        rule_pred["fold"] = fold

        predictions["rule_based"].append(rule_pred)

        # Logistic regression
        print("Running logistic regression...")

        logistic_pred = logistic_regression_detector(train, val)
        logistic_pred["fold"] = fold

        predictions["logistic_regression"].append(logistic_pred)
        
        # XGBoost
        print("Running XGBoost...")
        
        xgboost_pred = xgboost_detector(train, val)
        xgboost_pred["fold"] = fold
        
        predictions["xgboost"].append(xgboost_pred)

    # Aggregate and evaluate each method
    results = []
        
    for method, fold_predictions in predictions.items():

        # Skip methods that haven't been implemented yet
        if not fold_predictions:
            continue

        print(f"\n--- Evaluating {method} ---")

        # Combine out-of-fold predictions
        method_predictions = pd.concat(
            fold_predictions,
            ignore_index=True
        )

        # Convert timestamp predictions into meal events
        method_events = aggregate_predictions(
            method_predictions,
            threshold=0.5,
            smoothing_window_min=15,
            max_gap_min=15,
            max_duration_min=60
        )
        
        # One-to-one matching with reported meals
        matches, unmatched_predictions, unmatched_meals = match_events(
            method_events,
            meals,
            min_carbs=5,
            tolerance_min=30,
        )
        
        # Visualize detected meals
        plot_day(
            aligned,
            sid=14,
            date="2023-09-28",
            predicted_events=method_events,
            meals=meals,
            matches=matches,
            method=method,
            output_file=config.FIGURES_DIR / f"predictions_{method}_14_2023-09-28.png",
        )

        n_reported_meals = (meals["carbs"] >= 5).sum()

        recall = len(matches) / n_reported_meals

        median_signed_error = (
            matches["timing_error_min"].median()
            if not matches.empty else float("nan")
        )

        median_absolute_error = (
            matches["absolute_error_min"].median()
            if not matches.empty else float("nan")
        )

        results.append({
            "method": method,
            "reported_meals": n_reported_meals,
            "predicted_events": len(method_events),
            "matched_meals": len(matches),
            "unmatched_predictions": len(unmatched_predictions),
            "unmatched_meals": len(unmatched_meals),
            "recall": recall,
            "median_signed_error_min": median_signed_error,
            "median_absolute_error_min": median_absolute_error,
        })

        print(f"Reported meals ≥5 g: {n_reported_meals:,}")
        print(f"Predicted events: {len(method_events):,}")
        print(f"Matched meals: {len(matches):,}")
        print(f"Recall: {recall:.2%}")
        print(f"Median absolute timing error: {median_absolute_error:.1f} min")

    # Compare methods
    results_table = pd.DataFrame(results)

    print("\n--- Method comparison ---")
    print(results_table.to_string(index=False))

    results_table.to_csv(
        config.TABLES_DIR / "method_comparison.csv",
        index=False
    )


if __name__ == "__main__":
    main()