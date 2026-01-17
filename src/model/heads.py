from catboost import CatBoostClassifier
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier


def build_head(head_args: dict, random_state: int = 42) -> MultiOutputClassifier:
    head_type = head_args.get("type", "catboost").lower()
    logger.info(f"Head type: {head_type}")
    params = head_args.get(head_type) or {}

    classifiers = {
        "catboost": lambda p, rs: CatBoostClassifier(random_seed=rs, verbose=False, allow_writing_files=False, **p),
        "rf": lambda p, rs: RandomForestClassifier(random_state=rs, **p),
        "logreg": lambda p, rs: LogisticRegression(random_state=rs, **p),
    }

    try:
        builder = classifiers.get(head_type)
        if builder is None:
            raise ValueError(f"Unknown head type: {head_type}. Available types: {list(classifiers.keys())}")

        clf = builder(params, random_state)
        return MultiOutputClassifier(clf)

    except TypeError as e:
        logger.exception(f"Error creating head '{head_type}' with params {params}: {e}")
        raise
