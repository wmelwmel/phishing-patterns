from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from loguru import logger
from sklearn.manifold import TSNE
from tqdm import tqdm


def visualize_with_tsne_dbscan(
    vis_df: pd.DataFrame,
    perplexity: int = 30,
    metric: str = "cosine",
    title: str = "Embedding vizualization (t-SNE + DBSCAN)",
) -> None:
    df = vis_df.copy()

    clusters = np.array(df.cluster)
    unique_clusters = df.cluster.unique()
    n_clusters = len(unique_clusters)
    noise_count = (df.cluster == -1).sum()

    tsne = TSNE(n_components=2, perplexity=perplexity, metric=metric, random_state=42, init="random")
    coords = tsne.fit_transform(np.array(df.emb.to_list()))

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]
    df["cluster"] = df["cluster"].astype(str)
    df["is_noise"] = (clusters == -1).astype(str)

    df["symbol"] = df["label"].map({0: "square", 1: "x", -1: "circle"})

    fig = go.Figure()

    symbol_groups = ["circle", "square", "x"]
    cluster_groups = df["cluster"].unique()

    symbol_trace_indices: dict[str, list[int]] = {symbol: [] for symbol in symbol_groups}

    color_palette = px.colors.qualitative.Dark24
    cluster_colors = {str(c): color_palette[i % len(color_palette)] for i, c in enumerate(sorted(cluster_groups))}

    for cluster in tqdm(cluster_groups, desc="Cluster vizualization process"):
        cluster_str = str(cluster)
        cluster_color = cluster_colors[cluster_str]

        for symbol in symbol_groups:
            subset = df[(df["cluster"] == cluster_str) & (df["symbol"] == symbol)]
            if len(subset) == 0:
                continue

            trace = go.Scatter(
                x=subset["x"],
                y=subset["y"],
                mode="markers",
                marker=dict(
                    symbol=symbol,
                    size=4 if cluster_str == "-1" else 7,
                    color=cluster_color,
                    line=dict(width=0.5, color="DarkSlateGrey"),
                ),
                name=f"{cluster_str} ({symbol})",
                legendgroup=cluster_str,
                legendgrouptitle_text=f"Кластер {cluster_str}",
                visible=True,
                customdata=np.stack(
                    (subset["cluster"], subset["sentence"], subset["message_id"], subset["is_noise"], subset["label"]),
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>Cluster: %{customdata[0]}</b><br>"
                    "<b>Text:</b> %{customdata[1]}<br>"
                    "<b>Message ID:</b> %{customdata[2]}<br>"
                    "<b>Is noise:</b> %{customdata[3]}<br>"
                    "<b>Label:</b> %{customdata[4]}<br>"
                    "<extra></extra>"
                ),
            )
            fig.add_trace(trace)
            symbol_trace_indices[symbol].append(len(fig.data) - 1)

    buttons = [
        dict(label="All markers", method="update", args=[{"visible": [True] * len(fig.data)}]),
        dict(
            label="Only circles",
            method="update",
            args=[{"visible": [i in symbol_trace_indices["circle"] for i in range(len(fig.data))]}],
        ),
        dict(
            label="Only squares",
            method="update",
            args=[{"visible": [i in symbol_trace_indices["square"] for i in range(len(fig.data))]}],
        ),
        dict(
            label="Only Xs",
            method="update",
            args=[{"visible": [i in symbol_trace_indices["x"] for i in range(len(fig.data))]}],
        ),
    ]

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Clusters: {n_clusters} | Noise: {noise_count} ({noise_count / len(df):.1%})</sup>",
            x=0.05,
            xanchor="left",
        ),
        xaxis_title="t-SNE Dimension 1",
        yaxis_title="t-SNE Dimension 2",
        legend_title_text="Clusters",
        height=700,
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
        updatemenus=[
            dict(type="dropdown", direction="down", buttons=buttons, x=0.98, xanchor="right", y=1.15, yanchor="top")
        ],
        legend=dict(groupclick="toggleitem", itemsizing="constant", x=1.02, xanchor="left"),
    )

    fig.show()

    return df


def plot_bar_h(cluster_df: pd.DataFrame, top_n_clusters: int = 20, mode: str = "size") -> None:
    if mode == "size":
        target_x = "amount_percent"
        palette = "Reds_r"
        total_n = "total"
    elif mode == "phishing":
        target_x = "phishing_percent"
        palette = "Greens_d"
        total_n = "phishing_sum"

    cluster_stats = cluster_df.sort_values(target_x, ascending=False)

    if mode == "phishing":
        cluster_stats = cluster_stats[cluster_stats["phishing_percent"].between(0, 100, inclusive="neither")]

    top_clusters = cluster_stats.head(top_n_clusters)
    top_clusters["cluster"] = top_clusters["cluster"].astype(str)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    ax = sns.barplot(
        x=target_x,
        y="cluster",
        data=top_clusters,
        palette=palette,
        orient="h",
    )

    for p in ax.patches:
        width = p.get_width()
        plt.text(width + 0.5, p.get_y() + p.get_height() / 2.0, f"{width:.1f}%", ha="left", va="center")

    plt.title(f"Top {top_n_clusters} clusters ({mode})", fontsize=16)
    plt.xlabel("Percent, %", fontsize=12)
    plt.ylabel("Cluster number", fontsize=12)
    plt.xlim(0, 105)
    plt.tight_layout()

    for i, (_, row) in enumerate(top_clusters.iterrows()):
        plt.text(100, i, f"n={row[total_n]}", va="center", fontsize=9, color="gray")

    plt.show()


def get_cluster_info(emb_df: pd.DataFrame, clusters: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    emb_df = emb_df[["file_path", "label", "sentence", "emb", "lang"]]
    emb_df["cluster"] = clusters
    labels = np.array(emb_df.label)

    df_cluster = pd.DataFrame({"cluster": clusters, "is_phishing": [1 if label == 1 else 0 for label in labels]})

    cluster_stats = df_cluster.groupby("cluster")["is_phishing"].agg(total="count", phishing_sum="sum").reset_index()

    cluster_stats["phishing_percent"] = round((cluster_stats["phishing_sum"] / cluster_stats["total"]) * 100, 2)
    cluster_stats["amount_percent"] = round(cluster_stats["total"] / cluster_stats["total"].sum() * 100, 2)

    logger.info("Phishing percent per cluster statistics:")
    logger.info(f"  Max: {cluster_stats['phishing_percent'].max():.2f}%")
    logger.info(f"  Mean: {cluster_stats['phishing_percent'].mean():.2f}%")
    logger.info(f"  Median: {cluster_stats['phishing_percent'].median():.2f}%\n")

    logger.info("Fraction percent per cluster statistics:")
    logger.info(f"  Max: {cluster_stats['amount_percent'].max():.2f}%")
    logger.info(f"  Mean: {cluster_stats['amount_percent'].mean():.2f}%")
    logger.info(f"  Median: {cluster_stats['amount_percent'].median():.2f}%\n")

    clean_clusters = (cluster_stats.phishing_percent == 0).sum()
    phishing_clusters = (cluster_stats.phishing_percent == 100).sum()
    mixed_clusters = len(cluster_stats) - clean_clusters - phishing_clusters
    total_n_clusters = len(cluster_stats)

    logger.info("Cluster composition:")
    logger.info(f"  Clean: {clean_clusters} ({clean_clusters / total_n_clusters * 100:.2f}%)")
    logger.info(f"  Phishing: {phishing_clusters} ({phishing_clusters / total_n_clusters * 100:.2f}%)")
    logger.info(f"  Mixed: {mixed_clusters} ({mixed_clusters / total_n_clusters * 100:.2f}%)")

    plot_bar_h(cluster_stats, mode="size")
    plot_bar_h(cluster_stats, mode="phishing", top_n_clusters=40)

    return emb_df, cluster_stats


def get_relabel_statistic(
    df_results: pd.DataFrame, messages_id_vectors: dict[str, np.ndarray], features: list[dict]
) -> dict[str, dict[str, Any]]:
    total_files = len(df_results)

    not_labeled = df_results["res_vector"].isna().sum()
    not_labeled_pct = (not_labeled / total_files) * 100

    no_phishing = ((df_results["vector_sum"] == 0) & df_results["res_vector"].notna()).sum()

    vector_sum_stats = {
        "max": df_results["vector_sum"].max(),
        "mean": df_results["vector_sum"].mean(),
        "median": df_results["vector_sum"].median(),
    }

    file_stats_msg = (
        "File statistics:\n"
        f"  - Total files: {total_files} ({df_results['vector_sum'].sum()} triggers)\n"
        f"  - Excluded from labeling: {not_labeled} ({not_labeled_pct:.2f}%)\n"
        f"  - Files with no phishing features: {no_phishing}\n"
        f"  - Max features per file: {vector_sum_stats['max']}\n"
        f"  - Mean features per file: {vector_sum_stats['mean']:.2f}\n"
        f"  - Median features per file: {vector_sum_stats['median']}\n"
    )
    logger.info(file_stats_msg)

    feature_counts = np.zeros(len(features), dtype=int)
    for vec in messages_id_vectors.values():
        if vec is not None:
            feature_counts += vec.astype(int)

    features_df = pd.DataFrame(
        {"id": [f["id"] for f in features], "name": [f["name"] for f in features], "count": feature_counts}
    )

    features_df = features_df.sort_values("count", ascending=True)

    zero_features = features_df[features_df["count"] == 0]
    zero_features_count = len(zero_features)
    zero_features_pct = (zero_features_count / len(features)) * 100

    feature_stats_msg = (
        "Feature statistics:\n"
        f"  - Total unique features: {len(features)}\n"
        f"  - Zero-trigger features: {zero_features_count} ({zero_features_pct:.2f}%)"
    )
    logger.info(feature_stats_msg)

    plt.figure(figsize=(14, 10))
    plt.title("Feature trigger frequency", fontsize=16)

    colors = plt.cm.Blues(np.linspace(0.3, 1, len(features_df)))  # type: ignore[attr-defined]

    bars = plt.barh(features_df["name"], features_df["count"], color=colors, edgecolor="black")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + max(features_df["count"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(width)}",
            va="center",
            fontsize=10,
        )

    plt.xlabel("Number of triggers", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    if not zero_features.empty:
        logger.warning("Inactive features (zero triggers):")
        for _, row in zero_features.iterrows():
            logger.warning(f"- {row['name']} (ID: {row['id']})")

    plt.show()

    return {
        "file_stats": {
            "total_files": total_files,
            "not_labeled": not_labeled,
            "not_labeled_pct": not_labeled_pct,
            "no_phishing": no_phishing,
            "vector_sum_stats": vector_sum_stats,
        },
        "feature_stats": {
            "total_features": len(features),
            "zero_features_count": zero_features_count,
            "zero_features_pct": zero_features_pct,
            "features_df": features_df,
        },
    }


def plot_corr_matrix(feature_names: list[str], corr_matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=feature_names,
            y=feature_names,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Correlation"),
            hoverongaps=False,
            hoverinfo="x+y+z",
            hovertemplate=(
                "<b>Feature X:</b> %{x}<br><b>Feature Y:</b> %{y}<br><b>Correlation:</b> %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Feature correlation matrix",
        title_x=0.5,
        width=1000,
        height=1000,
        xaxis=dict(tickangle=90, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        hovermode="closest",
        margin=dict(l=50, r=50, b=150, t=100),
    )

    fig.update_layout(
        xaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="black", spikethickness=1),
        yaxis=dict(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="black", spikethickness=1),
    )

    fig.update_layout(dragmode="pan", hoverdistance=100, spikedistance=100)

    annotations = []
    for i, row in enumerate(corr_matrix.values):
        for j, value in enumerate(row):
            if abs(value) > 0.2 and i != j:
                annotations.append(
                    dict(
                        x=feature_names[j],
                        y=feature_names[i],
                        text=str(round(value, 2)),
                        showarrow=False,
                        font=dict(color="white" if abs(value) > 0.7 else "black"),
                    )
                )

    fig.update_layout(annotations=annotations)
    return fig
