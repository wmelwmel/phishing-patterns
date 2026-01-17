from pathlib import Path

root_dir = Path("/")
base_dir = Path(__file__).parent.parent
configs_dir = base_dir / "src/configs"

# Paths for loading and preprocessing initial data
mnt_data_dir = root_dir / "mnt/data/anti_phishing"
logs_dir = mnt_data_dir / "logs"
download_dir = mnt_data_dir / "s3_downloaded"

old_data_dir = download_dir / "old"
new_data_dir = download_dir / "new"

extract_dir = download_dir / "old_extracted"
clean_dir = extract_dir / "pass_infected/pass infected/Clean"
phishing_dir = extract_dir / "pass_infected/pass infected/Phishing"
new_ru_data_dir = mnt_data_dir / "new_ru_data"
new_data_labels_path = mnt_data_dir / "labels.csv"

eml_df_path = mnt_data_dir / "eml_df_upd.pkl"
lang_df_path = mnt_data_dir / "lang_df_upd.pkl"
sentences_dir = mnt_data_dir / "sentences_2608"

# Paths for clustering and labelling data
jinja_templates_dir = base_dir / "src/templates"
emb_df_path = mnt_data_dir / "df_emb_upd.pkl"
labeled_json_path = mnt_data_dir / "llm_labeled_results_newru.json"
labeled_df_path = mnt_data_dir / "labeled_results_df_oss_14k_newru.pkl"

llm_cluster_label_json = mnt_data_dir / "llm_cluster_label.json"
llm_synth_ds_json = mnt_data_dir / "llm_synth_ds.json"

manual_init_df_path = mnt_data_dir / "manual_init_df_FIXED.pkl"
synthetic_df_path = mnt_data_dir / "synthetic_df_300_upd.pkl"

mlflow_models_local_dir = mnt_data_dir / "mlflow_models"
