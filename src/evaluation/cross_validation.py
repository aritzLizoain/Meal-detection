from sklearn.model_selection import GroupKFold


def participant_cv(labeled, n_splits=5):
    """
    Generate participant-independent cross-validation folds.

    All timestamps belonging to one participant stay
    in the same fold.
    """
    cv = GroupKFold(n_splits=n_splits)

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(labeled, groups=labeled["sid"]),
        start=1
    ):
        train = labeled.iloc[train_idx].copy()
        val = labeled.iloc[val_idx].copy()

        yield fold, train, val