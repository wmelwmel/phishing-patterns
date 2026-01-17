import numpy as np
import pandas as pd
from loguru import logger
from skmultilearn.model_selection import IterativeStratification


class DataSplitter:
    def __init__(
        self, res_col: str, test_size: float, verbose: bool = True, strat_order: int = 2, random_seed: int = 42
    ) -> None:
        self.res_col = res_col
        self.test_size = test_size
        self.verbose = verbose
        self.strat_order = strat_order
        self.random_seed = random_seed

    def stratified_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        try:
            np.random.seed(self.random_seed)
            y_features = np.array(df[self.res_col].tolist())
            y_label = df["label"].values.reshape(-1, 1)
            y_lang = self._get_dummies(df["lang"])
            all_zero_flag = (y_features.sum(axis=1) == 0).astype(int).reshape(-1, 1)
            y_all = np.hstack([y_features, y_label, y_lang, all_zero_flag])

            stratifier = IterativeStratification(
                n_splits=2, order=self.strat_order, sample_distribution_per_fold=[self.test_size, 1 - self.test_size]
            )

            X_dummy = np.zeros((len(df), 1))
            train_idx, test_idx = next(stratifier.split(X_dummy, y_all))

            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            valid_features = self._get_valid_features(train_df)
            logger.info(f"Number of valid features: {len(valid_features)}")

            return train_df, test_df, valid_features
        except Exception as e:
            logger.error(f"Error during data splitting: {e}", exc_info=True)
            raise

    def _get_dummies(self, df_col: pd.Series) -> np.ndarray:
        return pd.get_dummies(df_col).values

    def _get_valid_features(self, df_check: pd.DataFrame) -> list[int]:
        y = np.array(df_check[self.res_col].tolist())

        valid_features = []
        for i in range(y.shape[1]):
            if 1 in y[:, i]:
                valid_features.append(i)
        return valid_features
