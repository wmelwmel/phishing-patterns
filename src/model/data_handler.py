from typing import Any, Literal

import numpy as np
import pandas as pd
from datasets import Dataset
from loguru import logger
from omegaconf import DictConfig
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.datamodels import DatasetsResult, LoadDFResult
from src.model.balancer import DataBalancer
from src.model.manual_data_handler import ManualDatasetHandler
from src.model.splitter import DataSplitter
from src.paths import labeled_df_path, manual_init_df_path, synthetic_df_path
from src.prepare_data.data_processing import normalize_text
from src.prepare_data.get_embeddings import encode_on_device, get_encoder_model
from src.utils import get_config, prepare_feature_info
from src.visualization import plot_corr_matrix


class DatasetHandler:
    def __init__(self, data_handler_config: DictConfig) -> None:
        self.get_synthetic_df: bool = data_handler_config.get_synthetic_df
        self.get_balanced_df: bool = data_handler_config.get_balanced_df
        self.balancer_mode: Literal["kmeans", "maxmin"] = data_handler_config.balancer_mode
        self.target_trigger_count: None | int = data_handler_config.target_trigger_count
        self.res_col: str = data_handler_config.res_col
        self.res_col_sent: str = data_handler_config.res_col_sent
        self.test_size: float = data_handler_config.test_size
        self.verbose: bool = data_handler_config.dh_verbose
        self.use_manual_ds: bool = data_handler_config.use_manual_ds
        self.encoder_name: str = data_handler_config.encoder_name
        self.sim_threshold: float = data_handler_config.sim_threshold
        self.strat_order: int = data_handler_config.strat_order
        self.indices_to_exclude, self.features = prepare_feature_info()

    def get_datasets(
        self,
    ) -> DatasetsResult:
        full_test_dataset, manual_eval_dataset, golden_dataset = None, None, None
        df_load_res = self._load_and_clean()
        init_labeled_dfs = df_load_res.labeled_df_filtered
        manual_eval_df = df_load_res.manual_eval_df
        llm_res_vec = df_load_res.llm_res_vec
        manual_res_vec = df_load_res.manual_res_vec

        init_labeled_df_concat = pd.concat(init_labeled_dfs)
        feature_names, corr_matrix = self._get_corr_matrix(init_labeled_df_concat)
        matrix_fig = plot_corr_matrix(feature_names, corr_matrix)

        labeled_df = self._prepare_labeled_df(init_labeled_dfs)
        train_df, test_df, valid_features = self._get_train_test(labeled_df)
        train_dataset, test_dataset = self._get_main_datasets(train_df, test_df)

        if self.get_balanced_df:
            full_test_dataset = self._get_full_test_dataset(init_labeled_df_concat, labeled_df, valid_features)

        if manual_eval_df is not None:
            manual_eval_dataset = self._get_manual_eval_dataset(manual_eval_df, valid_features)
            golden_dataset = self._get_golden_dataset(valid_features)

        return DatasetsResult(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            full_test_dataset=full_test_dataset,
            manual_eval_dataset=manual_eval_dataset,
            golden_dataset=golden_dataset,
            valid_features=valid_features,
            llm_res_vec=llm_res_vec,
            manual_res_vec=manual_res_vec,
            matrix_fig=matrix_fig,
        )

    def _load_and_clean(self) -> LoadDFResult:
        try:
            labeled_df = pd.read_pickle(labeled_df_path)
            labeled_df = self._filter_feature_columns(labeled_df)
            self._log_df_stats(labeled_df, "Initial labeled")

            manual_data = None
            if self.use_manual_ds:
                manual_data = self._load_manual_df(labeled_df)
                self._log_df_stats(manual_data.manual_eval_df, "Manual eval")

            labeled_df = self._clean_df(manual_data.labeled_df_filtered if manual_data else labeled_df)
            self._log_df_stats(pd.concat(labeled_df), "Cleaned labeled")

            return LoadDFResult(
                labeled_df_filtered=labeled_df,
                manual_eval_df=manual_data.manual_eval_df if manual_data else None,
                llm_res_vec=manual_data.llm_res_vec if manual_data else None,
                manual_res_vec=manual_data.manual_res_vec if manual_data else None,
            )
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            raise

    def _filter_feature_columns(self, df_init: pd.DataFrame) -> pd.DataFrame:
        if not self.indices_to_exclude:
            return df_init

        if df_init.empty:
            raise ValueError("Attempted to filter features from an empty DataFrame.")

        df = df_init.copy()
        cols_to_filter = [c for c in [self.res_col, self.res_col_sent] if c in df.columns]

        if not cols_to_filter:
            raise ValueError(f"Neither '{self.res_col}' nor '{self.res_col_sent}' columns found in DataFrame.")

        vec_len = self._validate_vector_columns(df, cols_to_filter)

        keep_mask = np.ones(vec_len, dtype=bool)
        keep_mask[self.indices_to_exclude] = False

        applied_cols = []
        for col in cols_to_filter:
            df[col] = df[col].apply(lambda v: self._apply_mask_to_value(v, keep_mask))
            applied_cols.append(col)

        if not applied_cols:
            raise ValueError(
                f"No valid feature columns were filtered in DataFrame (indices={self.indices_to_exclude})."
            )

        logger.info(f"Applied feature filtering (mask length={vec_len}) to columns: {applied_cols}")
        return df

    def _validate_vector_columns(self, df: pd.DataFrame, cols: list[str]) -> int:
        lengths = set()

        for col in cols:
            series = df[col].dropna()
            if series.empty:
                raise ValueError(f"Column '{col}' is empty.")

            first_val = series.iloc[0]
            if self._vec_format_check(first_val):
                first_elem = first_val[0] if len(first_val) > 0 else None

                if self._vec_format_check(first_elem):
                    inner_lengths = set()
                    for inner in first_val:
                        if self._vec_format_check(inner):
                            inner_lengths.add(len(inner))
                        else:
                            msg = (
                                f"Invalid inner data type in column '{col}': "
                                f"expected list or ndarray, got {type(inner)}"
                            )
                            raise ValueError(msg)

                    if len(inner_lengths) != 1:
                        raise ValueError(f"Inconsistent inner vector lengths in column '{col}'.")
                    lengths |= inner_lengths
                else:
                    lengths.add(len(first_val))
            else:
                raise ValueError(
                    f"Invalid data type in column '{col}': expected list or ndarray. Got: {type(first_val)}"
                )

        if len(lengths) != 1:
            raise ValueError(f"Vector length mismatch across columns {cols}: found {lengths}.")

        return lengths.pop()

    def _apply_mask_to_value(self, val: Any, keep_mask: np.ndarray) -> Any:
        if self._vec_format_check(val):
            if len(val) == 0:
                return val

            first_elem = val[0]
            if self._vec_format_check(first_elem):
                filtered = []
                for inner in val:
                    if self._vec_format_check(inner) and len(inner) == len(keep_mask):
                        filtered.append(np.asarray(inner)[keep_mask].tolist())
                    else:
                        filtered.append(inner)
                return filtered
            else:
                if len(val) == len(keep_mask):
                    return np.asarray(val)[keep_mask].tolist()
        return val

    def _vec_format_check(self, vec: Any) -> bool:
        return isinstance(vec, (list, np.ndarray))

    def _log_df_stats(self, df: pd.DataFrame, name: str, top_n: int = 5) -> None:
        ds_name = f"{name} dataset"
        logger.info(f"{ds_name}:")
        logger.info(f"Shape: {df.shape}")
        logger.info("Label distribution:\n" + df.label.value_counts(normalize=True).to_string())
        logger.info(
            f"Language distribution ({ds_name} top_{top_n}):\n"
            + df.lang.value_counts(normalize=True).head(top_n).to_string()
        )

    def _load_manual_df(
        self, labeled_df: pd.DataFrame, manual_labels_name: str = "email_patterns_manual_labels.yaml"
    ) -> LoadDFResult:
        manual_ds_handler = ManualDatasetHandler(self.features)
        return manual_ds_handler.get_manual_eval_df(labeled_df, manual_labels_name)

    def _clean_df(self, df: pd.DataFrame) -> list[pd.DataFrame]:
        clean_df, languages = self._filter_valid_langs(df)
        clean_dfs = self._drop_implicit_duplicates(clean_df, languages)
        return clean_dfs

    def _filter_valid_langs(self, df: pd.DataFrame) -> pd.DataFrame:
        clean_df = df.copy()
        clean_df = clean_df[clean_df.res_vector.notna()]
        languages = get_config("lang_config.yaml")["languages"]
        if languages:
            clean_df = clean_df[clean_df.lang.isin(languages)]
        return clean_df, languages

    def _drop_implicit_duplicates(self, df: pd.DataFrame, languages: list[str]) -> list[pd.DataFrame]:
        labeled_df_exp = df.explode(["sentence", self.res_col_sent])
        if self.get_synthetic_df:
            synth_df = pd.read_pickle(synthetic_df_path)
            synth_df = self._filter_feature_columns(synth_df)
            logger.info(f"Downloaded synthetic dataset from: {synthetic_df_path} ({len(synth_df)} rows)")
            labeled_df_exp = pd.concat([labeled_df_exp, synth_df])
        labeled_df_exp["normalized"] = labeled_df_exp["sentence"].apply(normalize_text)
        labeled_df_exp_unique = labeled_df_exp.drop_duplicates("normalized")
        init_labeled_files_c = labeled_df_exp.file_path.nunique()
        clean_labeled_files_c = labeled_df_exp_unique.file_path.nunique()
        diff_files_c = init_labeled_files_c - clean_labeled_files_c

        logger.info(
            f"Files filtered (after normalization): {diff_files_c} ({diff_files_c / init_labeled_files_c * 100:.2f}%)"
        )
        logger.info(f"{init_labeled_files_c} -> {clean_labeled_files_c}")

        lang_dfs = self._get_filtered_lang_dfs(labeled_df_exp_unique, languages)

        return lang_dfs

    def _get_filtered_lang_dfs(
        self,
        labeled_df_exp: pd.DataFrame,
        langs: list[str],
    ) -> list[pd.DataFrame]:
        lang_dfs = []
        st_model = get_encoder_model(sentence_encoder_name=self.encoder_name)

        for lang in langs:
            lang_labeled_df_exp = labeled_df_exp[labeled_df_exp.lang == lang]
            lang_embeddings = encode_on_device(lang_labeled_df_exp["normalized"].to_list(), st_model)
            lang_labeled_df_exp["emb"] = lang_embeddings.tolist()
            logger.info(f"Filter implicit duplicates ({lang} df)...")
            filtered_lang_df = self._filter_implicit_dup(lang_labeled_df_exp, lang_embeddings, self.sim_threshold)
            logger.info(
                f"Filtered {len(lang_labeled_df_exp) - len(filtered_lang_df)} sentences"
                f"(out of {len(lang_labeled_df_exp)})"
            )
            lang_dfs.append(filtered_lang_df)

        return lang_dfs

    def _filter_implicit_dup(self, df: pd.DataFrame, lang_embs: np.ndarray, sim_threshold: float) -> pd.DataFrame:
        filtered_df = df.copy()
        count_n = len(filtered_df)

        nbrs = NearestNeighbors(radius=1 - sim_threshold, metric="cosine", n_jobs=-1).fit(lang_embs)
        neighbors_list = nbrs.radius_neighbors(lang_embs, return_distance=False)
        mask = np.ones(count_n, dtype=bool)

        for i in tqdm(range(count_n)):
            if mask[i]:
                duplicates = neighbors_list[i][neighbors_list[i] != i]
                mask[duplicates] = False

        return filtered_df[mask]

    def _get_corr_matrix(self, init_labeled_df: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
        logger.info("Calculate correlation matrix...")
        feature_names = [f"{feat['id']}_{feat['name']}" for feat in self.features]
        feature_matrix = pd.DataFrame(init_labeled_df[self.res_col_sent].to_list(), columns=feature_names)
        corr_matrix = feature_matrix.corr(method="pearson")
        return feature_names, corr_matrix

    def _prepare_labeled_df(self, labeled_df: list[pd.DataFrame]) -> pd.DataFrame:
        if self.get_balanced_df:
            data_balancer = DataBalancer(
                self.res_col_sent, self.target_trigger_count, self.verbose, mode=self.balancer_mode
            )
            balanced_df = data_balancer.get_balanced_df(labeled_df)
            self._log_df_stats(balanced_df, "Balanced")
            return balanced_df
        return pd.concat(labeled_df)

    def _get_train_test(self, labeled_df_exp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
        data_splitter = DataSplitter(self.res_col_sent, self.test_size, strat_order=self.strat_order)
        train_df, test_df, valid_features = data_splitter.stratified_split(labeled_df_exp)
        train_df = self._filter_features(train_df, valid_features, filter_sent_vec=True)

        test_df = test_df.groupby("file_path").apply(self._aggregate_group).reset_index()
        test_df = self._filter_features(test_df, valid_features)
        if self.verbose:
            self._log_stats(train_df, test_df, valid_features)
        return train_df, test_df, valid_features

    def _filter_features(
        self, df: pd.DataFrame, valid_features: list[int], filter_sent_vec: bool = False
    ) -> pd.DataFrame:
        df = df.assign(res_vector_filtered=df[self.res_col].apply(lambda x: [x[i] for i in valid_features]))

        if filter_sent_vec:
            df = df.assign(res_vec_sent_filtered=df[self.res_col_sent].apply(lambda x: [x[i] for i in valid_features]))
        return df

    def _aggregate_group(self, group: pd.Series) -> pd.Series:
        sentences = group["sentence"].tolist()
        vectors = group[self.res_col_sent].tolist()
        sum_vector = np.array(vectors).sum(axis=0) > 0

        group_lang = group["lang"]
        group_label = group["label"]

        if group_lang.nunique() > 1:
            logger.warning(f"Different lang values in group: {group['file_path'].iloc[0]}")
        if group_label.nunique() > 1:
            logger.warning(f"Different label values in group: {group['file_path'].iloc[0]}")

        return pd.Series(
            {
                "sentence": sentences,
                self.res_col_sent: vectors,
                self.res_col: sum_vector.astype(int),
                "lang": group_lang.iloc[0],
                "label": group_label.iloc[0],
            }
        )

    def _log_stats(self, train_df: pd.DataFrame, test_df: pd.DataFrame, valid_features: list[int]) -> None:
        self._log_df_stats(train_df, "Training")
        self._log_df_stats(test_df, "Testing")
        self._log_splitted_stats(valid_features, train_df, test_df.explode(self.res_col_sent))

    def _log_splitted_stats(self, valid_features: list[int], train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        for valid_feature_idx, feature_idx in enumerate(valid_features):
            feature_name = self.features[feature_idx]["name"]
            train_vals = train_df["res_vec_sent_filtered"].apply(lambda x: x[valid_feature_idx])
            test_vals = test_df[self.res_col_sent].apply(lambda x: x[valid_feature_idx])
            logger.info(f"Feature {feature_name}:")
            logger.info(
                f"  Train: {train_vals.sum()}/{len(train_vals)} ({train_vals.sum() / len(train_vals) * 100:.2f}%)"
            )
            logger.info(f"  Test: {test_vals.sum()}/{len(test_vals)} ({test_vals.sum() / len(test_vals) * 100:.2f}%)")

        train_zero_count = (train_df[self.res_col_sent].apply(sum) == 0).sum()
        test_zero_count = (test_df[self.res_col_sent].apply(sum) == 0).sum()
        total_train = len(train_df)
        total_test = len(test_df)

        logger.info("Zero-vector samples:")
        logger.info(f"  Train: {train_zero_count}/{total_train} ({train_zero_count / total_train * 100:.2f}%)")
        logger.info(f"  Test: {test_zero_count}/{total_test} ({test_zero_count / total_test * 100:.2f}%)")

    def _get_main_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[Dataset, Dataset]:
        return self._get_dataset(train_df, "res_vec_sent_filtered"), self._get_dataset(test_df)

    def _get_dataset(self, df: pd.DataFrame, res_col: str = "res_vector_filtered") -> Dataset:
        return Dataset.from_dict(
            {
                "text": df["sentence"].tolist(),
                "label": df[res_col].tolist(),
                "lang": df["lang"].tolist(),
                "file_path": df["file_path"].tolist(),
            }
        )

    def _get_full_test_dataset(
        self, init_labeled_df: pd.DataFrame, labeled_df: pd.DataFrame, valid_features: list[int]
    ) -> Dataset | None:
        init_labeled_df_grouped = init_labeled_df.groupby("file_path").apply(self._aggregate_group).reset_index()
        selected_files = labeled_df.file_path.unique()
        full_test_df = init_labeled_df_grouped[~init_labeled_df_grouped.file_path.isin(selected_files)]
        if full_test_df.empty:
            return None
        full_test_df = self._filter_features(full_test_df, valid_features)
        self._log_df_stats(full_test_df, "Full test_df")
        return self._get_dataset(full_test_df)

    def _get_manual_eval_dataset(self, manual_eval_df: pd.DataFrame, valid_features: list[int]) -> Dataset:
        manual_eval_df = self._filter_features(manual_eval_df, valid_features)
        return self._get_dataset(manual_eval_df)

    def _get_golden_dataset(self, valid_features: list[int]) -> Dataset:
        golden_df = pd.read_pickle(manual_init_df_path)
        golden_df = self._filter_feature_columns(golden_df)
        golden_df = self._filter_features(golden_df, valid_features)
        self._log_df_stats(golden_df, "Full manual_init_df")
        return self._get_dataset(golden_df)
