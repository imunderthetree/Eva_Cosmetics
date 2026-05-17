from __future__ import annotations

import argparse
import json
import math
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
    module="joblib.externals.loky.backend.context",
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "eva_group_emails.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "employee_raise"

RANDOM_STATE = 42
N_TOPICS = 20
RAISE_THRESHOLD = 0.65
REVIEW_THRESHOLD = 0.45
GAP_THRESHOLD = 0.30

REQUIRED_COLUMNS = [
    "email_id",
    "from_name",
    "subject",
    "body",
    "skill_gap",
    "behavior_flag",
]

EXCLUDED_EMPLOYEES = {
    "All Department Heads",
    "Dr. Youssef Gamal",
    "Dr. Amr Wahba",
}

STOPWORDS = {
    "i",
    "me",
    "my",
    "we",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "it",
    "its",
    "they",
    "them",
    "their",
    "what",
    "which",
    "who",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "as",
    "of",
    "at",
    "by",
    "for",
    "with",
    "to",
    "from",
    "in",
    "on",
    "up",
    "so",
    "no",
    "not",
    "can",
    "will",
    "just",
    "now",
    "may",
    "please",
    "let",
    "know",
    "would",
    "could",
    "also",
    "get",
    "like",
    "well",
    "hi",
    "dear",
    "regards",
    "best",
    "thanks",
    "thank",
    "email",
    "team",
    "shall",
    "going",
    "sure",
    "need",
}

LEADERSHIP_KEYWORDS = {
    "Communication": ["communicate", "transparent", "clarity", "stakeholder", "feedback", "respond"],
    "Accountability": ["accountable", "responsible", "ownership", "integrity", "compliance", "deliver"],
    "Collaboration": ["collaborate", "coordinate", "support", "share", "align", "partner"],
    "Strategic Thinking": ["strategy", "initiative", "vision", "priority", "risk", "analysis", "forecast"],
    "Mentorship": ["mentor", "coach", "develop", "train", "talent", "empower", "guidance"],
    "Problem Solving": ["solve", "solution", "escalate", "investigate", "improve", "address", "root"],
}

SKILL_COLUMNS = list(LEADERSHIP_KEYWORDS)
DECISION_COLORS = {
    "Raise Recommended": "#27ae60",
    "Under Review": "#e67e22",
    "Not Recommended": "#c0392b",
}


@dataclass(frozen=True)
class PipelineResult:
    employees: pd.DataFrame
    decisions: pd.DataFrame
    model_name: str
    cv_results: dict[str, dict[str, float]]
    feature_names: list[str]
    output_paths: dict[str, str]


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [token for token in text.split() if token not in STOPWORDS and len(token) > 2]
    return " ".join(tokens)


def encode_flag(flag: Any) -> float:
    if not isinstance(flag, str):
        return 0.0
    if flag.startswith("Positive"):
        if "Exceptional" in flag or "Brave" in flag or "Exemplary" in flag:
            return 1.2
        return 1.0
    if flag.startswith("Negative"):
        if "Abuse" in flag or "Recurring" in flag or "Culture" in flag:
            return -1.4
        return -1.0
    return 0.0


def decide(score: float) -> str:
    if score >= RAISE_THRESHOLD:
        return "Raise Recommended"
    if score >= REVIEW_THRESHOLD:
        return "Under Review"
    return "Not Recommended"


def validate_columns(frame: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in employee email CSV: {missing_columns}")


def load_emails(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Employee email CSV not found: {data_path}")
    frame = pd.read_csv(data_path)
    validate_columns(frame)
    return frame


def filter_employees(
    frame: pd.DataFrame,
    min_emails: int = 5,
    excluded_employees: set[str] = EXCLUDED_EMPLOYEES,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    counts = frame["from_name"].value_counts()
    employees = [
        employee
        for employee in counts[counts >= min_emails].index
        if employee not in excluded_employees
    ]
    if not employees:
        raise ValueError(f"No employees have at least {min_emails} emails after exclusions.")
    filtered = frame[frame["from_name"].isin(employees)].copy().reset_index(drop=True)
    return filtered, counts, employees


def count_keyword_hits(text: str, keywords: list[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def add_email_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["clean_body"] = frame["body"].map(clean_text)
    frame["clean_subject"] = frame["subject"].map(clean_text)
    frame["clean_text"] = frame["clean_subject"] + " " + frame["clean_body"]
    frame["flag_score"] = frame["behavior_flag"].map(encode_flag)
    frame["has_gap"] = frame["skill_gap"].notna().astype(float) * -0.3
    frame["body_len"] = frame["body"].fillna("").map(len)
    frame["token_count"] = frame["clean_text"].map(lambda text: len(text.split()))

    for skill, keywords in LEADERSHIP_KEYWORDS.items():
        frame[skill] = frame["clean_text"].map(lambda text, kws=keywords: count_keyword_hits(text, kws))

    return frame


def build_skill_coverage(frame: pd.DataFrame, employees: list[str]) -> tuple[pd.DataFrame, list[str]]:
    inverted_index: defaultdict[str, defaultdict[str, set[Any]]] = defaultdict(lambda: defaultdict(set))
    use_email_id = "email_id" in frame.columns

    for index, row in frame.iterrows():
        employee = row["from_name"]
        email_id = row["email_id"] if use_email_id else index
        for token in row["clean_text"].split():
            inverted_index[token][employee].add(email_id)

    rows = []
    for employee in employees:
        row: dict[str, Any] = {"from_name": employee}
        for skill, tokens in LEADERSHIP_KEYWORDS.items():
            matched = [token for token in tokens if employee in inverted_index.get(token, {})]
            row[f"coverage_{skill}"] = len(matched) / len(tokens) if tokens else 0.0
            row[f"{skill}_evidence"] = matched
        rows.append(row)

    coverage = pd.DataFrame(rows)
    coverage_columns = [f"coverage_{skill}" for skill in SKILL_COLUMNS]
    return coverage, coverage_columns


def add_lsa_features(frame: pd.DataFrame, n_topics: int = N_TOPICS) -> tuple[pd.DataFrame, list[str], float]:
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(frame["clean_text"])
    max_components = min(n_topics, max(1, tfidf_matrix.shape[0] - 1), max(1, tfidf_matrix.shape[1] - 1))

    if max_components < 1:
        return frame.copy(), [], 0.0

    svd = TruncatedSVD(n_components=max_components, random_state=RANDOM_STATE)
    lsa_matrix = svd.fit_transform(tfidf_matrix)
    lsa_columns = [f"lsa_{index}" for index in range(max_components)]

    frame = frame.copy()
    frame[lsa_columns] = lsa_matrix
    explained_variance = float(svd.explained_variance_ratio_.sum())
    return frame, lsa_columns, explained_variance


def build_employee_matrix(
    frame: pd.DataFrame,
    counts: pd.Series,
    coverage: pd.DataFrame,
    coverage_columns: list[str],
    lsa_columns: list[str],
) -> pd.DataFrame:
    aggregate_columns = ["flag_score", "has_gap", "body_len", "token_count"] + SKILL_COLUMNS + lsa_columns
    employee_frame = frame.groupby("from_name")[aggregate_columns].mean().reset_index()
    employee_frame["email_count"] = employee_frame["from_name"].map(counts)
    employee_frame["log_email_count"] = np.log1p(employee_frame["email_count"])
    max_log_count = employee_frame["log_email_count"].max() or 1.0
    employee_frame["log_email_count"] = employee_frame["log_email_count"] / max_log_count
    employee_frame = employee_frame.merge(coverage, on="from_name", how="left")
    employee_frame[coverage_columns] = employee_frame[coverage_columns].fillna(0.0)
    return employee_frame


def add_target_scores(employee_frame: pd.DataFrame) -> pd.DataFrame:
    employee_frame = employee_frame.copy()
    cluster_features = ["flag_score"] + SKILL_COLUMNS + ["log_email_count"]
    scaled = StandardScaler().fit_transform(employee_frame[cluster_features].values)

    n_clusters = min(3, len(employee_frame))
    if n_clusters < 2:
        employee_frame["cluster"] = 0
        employee_frame["target_score"] = 1.0
        return employee_frame

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
    employee_frame["cluster"] = kmeans.fit_predict(scaled)

    rank_denominator = max(1, n_clusters - 1)
    cluster_rank = employee_frame.groupby("cluster")["flag_score"].mean().rank(ascending=True)
    cluster_base = {cluster: (rank - 1) / rank_denominator for cluster, rank in cluster_rank.items()}

    def within_cluster_percentile(row: pd.Series) -> float:
        members = employee_frame[employee_frame["cluster"] == row["cluster"]]
        return float((members["flag_score"] <= row["flag_score"]).mean() * 0.4)

    employee_frame["cluster_base"] = employee_frame["cluster"].map(cluster_base)
    employee_frame["within_pct"] = employee_frame.apply(within_cluster_percentile, axis=1)
    employee_frame["raw_target"] = employee_frame["cluster_base"] * 0.6 + employee_frame["within_pct"] * 0.4
    employee_frame["target_score"] = MinMaxScaler().fit_transform(employee_frame[["raw_target"]]).flatten()
    return employee_frame


def candidate_models() -> dict[str, Any]:
    return {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "Ridge Regression": Ridge(alpha=1.0),
    }


def evaluate_models(models: dict[str, Any], features: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    if len(target) < 3:
        return {name: {"r2": float("nan"), "rmse": float("nan")} for name in models}

    n_splits = min(5, len(target))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict[str, float]] = {}

    for name, model in models.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2_scores = cross_val_score(model, features, target, cv=cv, scoring="r2")
            mse_scores = cross_val_score(model, features, target, cv=cv, scoring="neg_mean_squared_error")
        r2_mean = float(np.nanmean(r2_scores))
        rmse_mean = float(np.sqrt(-np.nanmean(mse_scores)))
        results[name] = {"r2": r2_mean, "rmse": rmse_mean}

    return results


def choose_best_model(results: dict[str, dict[str, float]]) -> str:
    def sort_key(item: tuple[str, dict[str, float]]) -> tuple[float, float]:
        scores = item[1]
        r2 = scores["r2"]
        if math.isnan(r2):
            r2 = -math.inf
        return r2, -scores["rmse"]

    return max(results.items(), key=sort_key)[0]


def train_and_score(employee_frame: pd.DataFrame, coverage_columns: list[str], lsa_columns: list[str]) -> tuple[pd.DataFrame, str, dict[str, dict[str, float]], list[str], Any]:
    employee_frame = employee_frame.copy()
    feature_names = ["flag_score", "has_gap", "log_email_count", "token_count"] + SKILL_COLUMNS + coverage_columns + lsa_columns
    features = employee_frame[feature_names].values
    target = employee_frame["target_score"].values
    scaled_features = StandardScaler().fit_transform(features)

    models = candidate_models()
    cv_results = evaluate_models(models, scaled_features, target)
    best_name = choose_best_model(cv_results)
    best_model = models[best_name]
    best_model.fit(scaled_features, target)

    raw_scores = np.clip(best_model.predict(scaled_features), 0.0, 1.0)
    employee_frame["ml_score_raw"] = raw_scores
    employee_frame["ml_score"] = MinMaxScaler().fit_transform(employee_frame[["ml_score_raw"]]).flatten()
    employee_frame["decision"] = employee_frame["ml_score"].map(decide)

    return employee_frame, best_name, cv_results, feature_names, best_model


def add_radar_scores(employee_frame: pd.DataFrame, feature_names: list[str], model: Any) -> pd.DataFrame:
    employee_frame = employee_frame.copy()
    feature_values = employee_frame[feature_names].values
    scaled_features = StandardScaler().fit_transform(feature_values)
    scaled_frame = pd.DataFrame(scaled_features, columns=feature_names)

    skill_predictions: dict[str, np.ndarray] = {}
    for skill in SKILL_COLUMNS:
        skill_frame = scaled_frame.copy()
        other_skills = [column for column in SKILL_COLUMNS if column != skill]
        skill_frame[other_skills] = 0.0
        skill_predictions[skill] = model.predict(skill_frame.values)

    skill_matrix = np.column_stack([skill_predictions[skill] for skill in SKILL_COLUMNS])
    skill_matrix_norm = MinMaxScaler().fit_transform(skill_matrix)
    for index, skill in enumerate(SKILL_COLUMNS):
        employee_frame[f"radar_{skill}"] = skill_matrix_norm[:, index]

    return employee_frame


def decisions_table(employee_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["from_name", "email_count", "ml_score", "decision"] + SKILL_COLUMNS
    return employee_frame[columns].sort_values("ml_score", ascending=False).reset_index(drop=True)


def save_feature_importance_plot(
    output_dir: Path,
    model: Any,
    model_name: str,
    feature_names: list[str],
) -> Path | None:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        return None

    importance_frame = pd.DataFrame({"feature": feature_names, "importance": importances})
    non_lsa = importance_frame[~importance_frame["feature"].str.startswith("lsa_")]
    top_non_lsa = non_lsa.sort_values("importance", ascending=False).head(15)
    top_lsa = importance_frame[importance_frame["feature"].str.startswith("lsa_")].sort_values("importance", ascending=False).head(5)
    top_importance = pd.concat([top_non_lsa, top_lsa]).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if not feature.startswith("lsa_") else "#3498db" for feature in top_importance["feature"]]
    ax.barh(top_importance["feature"], top_importance["importance"], color=colors, edgecolor="white")
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Feature importance - {model_name}", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(
        handles=[
            mpatches.Patch(color="#2ecc71", label="Structured feature"),
            mpatches.Patch(color="#3498db", label="LSA text topic"),
        ],
        fontsize=9,
    )
    fig.tight_layout()

    path = output_dir / "feature_importance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_raise_bar_plot(output_dir: Path, decisions: pd.DataFrame, model_name: str) -> Path:
    colors = decisions["decision"].map(DECISION_COLORS)
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(
        decisions["from_name"],
        decisions["ml_score"],
        color=colors,
        edgecolor="white",
        linewidth=0.8,
        height=0.7,
    )

    ax.axvline(RAISE_THRESHOLD, color="#27ae60", linewidth=2, linestyle="--", label=f"Raise threshold ({RAISE_THRESHOLD})")
    ax.axvline(REVIEW_THRESHOLD, color="#e67e22", linewidth=2, linestyle="--", label=f"Review threshold ({REVIEW_THRESHOLD})")

    for bar, score in zip(bars, decisions["ml_score"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2, f"{score:.4f}", va="center", fontsize=8.5, fontweight="bold")

    patches = [mpatches.Patch(color=color, label=label) for label, color in DECISION_COLORS.items()]
    threshold_handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=patches + threshold_handles, loc="lower right", fontsize=9)

    ax.set_xlabel("ML performance score (0-1)", fontsize=11)
    ax.set_title(f"EVA Group raise decision via {model_name}", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = output_dir / "raise_decision_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_radar(ax: Any, values: list[float], employee: str, decision: str, color: str) -> None:
    angles = [index / len(SKILL_COLUMNS) * 2 * math.pi for index in range(len(SKILL_COLUMNS))]
    angles += angles[:1]
    values = values + values[:1]

    for radius in [0.25, 0.5, 0.75, 1.0]:
        xs = [radius * math.cos(angle) for angle in np.linspace(0, 2 * math.pi, 120)]
        ys = [radius * math.sin(angle) for angle in np.linspace(0, 2 * math.pi, 120)]
        ax.plot(xs, ys, color="#dddddd", linewidth=0.6, zorder=1)

    for angle in angles[:-1]:
        ax.plot([0, math.cos(angle)], [0, math.sin(angle)], color="#dddddd", linewidth=0.8, zorder=1)

    xs = [value * math.cos(angle) for value, angle in zip(values, angles)]
    ys = [value * math.sin(angle) for value, angle in zip(values, angles)]
    ax.fill(xs, ys, alpha=0.28, color=color, zorder=2)
    ax.plot(xs, ys, color=color, linewidth=2.0, zorder=3)
    ax.scatter(xs, ys, color=color, s=28, zorder=4)

    short_labels = {
        "Communication": "Comm.",
        "Accountability": "Account.",
        "Collaboration": "Collab.",
        "Strategic Thinking": "Strategy",
        "Mentorship": "Mentor",
        "Problem Solving": "Problem\nSolving",
    }
    for label, angle, value in zip(SKILL_COLUMNS, angles[:-1], values[:-1]):
        ax.text(
            1.28 * math.cos(angle),
            1.28 * math.sin(angle),
            short_labels.get(label, label),
            ha="center",
            va="center",
            fontsize=6.2,
            fontweight="bold",
            color="#333333",
        )
        ax.text(
            (value + 0.14) * math.cos(angle),
            (value + 0.14) * math.sin(angle),
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=5.2,
            color=color,
            fontweight="bold",
        )

    ax.set_title(f"{employee}\n{decision}", fontsize=7.5, fontweight="bold", pad=4, color="#1a1a2e")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)


def save_radar_plot(output_dir: Path, employee_frame: pd.DataFrame, model_name: str) -> Path:
    sorted_employees = employee_frame.sort_values("ml_score", ascending=False).reset_index(drop=True)
    employee_count = len(sorted_employees)
    columns = 5
    rows = math.ceil(employee_count / columns)

    fig = plt.figure(figsize=(columns * 4, rows * 4))
    grid = gridspec.GridSpec(rows, columns, figure=fig)
    axes = [fig.add_subplot(grid[index // columns, index % columns]) for index in range(rows * columns)]

    for index, row in sorted_employees.iterrows():
        values = [float(row[f"radar_{skill}"]) for skill in SKILL_COLUMNS]
        draw_radar(axes[index], values, row["from_name"], row["decision"], DECISION_COLORS[row["decision"]])

    for index in range(employee_count, len(axes)):
        axes[index].axis("off")

    legend_patches = [mpatches.Patch(color=color, label=label) for label, color in DECISION_COLORS.items()]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(
        f"EVA Group ML leadership skill radar charts ({model_name})",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    path = output_dir / "leadership_radar_charts.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def build_gap_report(employee_frame: pd.DataFrame) -> str:
    lines: list[str] = []
    for _, row in employee_frame.sort_values("ml_score", ascending=False).iterrows():
        lines.append(f"Employee : {row['from_name']}")
        lines.append(f"ML Score : {row['ml_score']:.3f} -> {row['decision']}")
        for skill in SKILL_COLUMNS:
            score = float(row[f"coverage_{skill}"])
            evidence = row[f"{skill}_evidence"]
            bar = "#" * int(score * 10) + "." * (10 - int(score * 10))
            flag = "GAP" if score < GAP_THRESHOLD else "Partial" if score < 0.60 else "OK"
            lines.append(f"  {skill:<22} {bar} {score:.2f} {flag}")
            if score < GAP_THRESHOLD:
                lines.append(f"    No hits for: {', '.join(LEADERSHIP_KEYWORDS[skill])}")
                lines.append("    Development plan recommended")
            elif evidence:
                lines.append(f"    Evidence tokens: {', '.join(evidence)}")
        lines.append("")
    return "\n".join(lines)


def save_outputs(
    output_dir: Path,
    employee_frame: pd.DataFrame,
    decisions: pd.DataFrame,
    model_name: str,
    cv_results: dict[str, dict[str, float]],
    feature_names: list[str],
    model: Any,
    save_plots: bool,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}

    decisions_path = output_dir / "raise_decisions.csv"
    decisions.to_csv(decisions_path, index=False)
    output_paths["decisions"] = str(decisions_path)

    gap_report_path = output_dir / "skill_gap_report.txt"
    gap_report_path.write_text(build_gap_report(employee_frame), encoding="utf-8")
    output_paths["gap_report"] = str(gap_report_path)

    summary_payload = {
        "model_name": model_name,
        "cv_results": cv_results,
        "employee_count": int(len(employee_frame)),
        "feature_count": int(len(feature_names)),
        "raise_threshold": RAISE_THRESHOLD,
        "review_threshold": REVIEW_THRESHOLD,
        "decision_counts": decisions["decision"].value_counts().to_dict(),
    }
    summary_path = output_dir / "model_summary.json"
    summary_path.write_text(json.dumps(summary_payload, ensure_ascii=True, indent=2), encoding="utf-8")
    output_paths["summary"] = str(summary_path)

    if save_plots:
        feature_plot = save_feature_importance_plot(output_dir, model, model_name, feature_names)
        if feature_plot is not None:
            output_paths["feature_importance"] = str(feature_plot)
        output_paths["raise_bar"] = str(save_raise_bar_plot(output_dir, decisions, model_name))
        output_paths["radar"] = str(save_radar_plot(output_dir, employee_frame, model_name))

    return output_paths


def run_pipeline(
    data_path: Path = DATA_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_emails: int = 5,
    save_plots: bool = True,
) -> PipelineResult:
    np.random.seed(RANDOM_STATE)

    emails = load_emails(data_path)
    emails, counts, employees = filter_employees(emails, min_emails=min_emails)
    emails = add_email_features(emails)
    coverage, coverage_columns = build_skill_coverage(emails, employees)
    emails, lsa_columns, _ = add_lsa_features(emails)
    employee_frame = build_employee_matrix(emails, counts, coverage, coverage_columns, lsa_columns)
    employee_frame = add_target_scores(employee_frame)
    employee_frame, model_name, cv_results, feature_names, model = train_and_score(employee_frame, coverage_columns, lsa_columns)
    employee_frame = add_radar_scores(employee_frame, feature_names, model)
    decisions = decisions_table(employee_frame)
    output_paths = save_outputs(output_dir, employee_frame, decisions, model_name, cv_results, feature_names, model, save_plots)

    return PipelineResult(
        employees=employee_frame,
        decisions=decisions,
        model_name=model_name,
        cv_results=cv_results,
        feature_names=feature_names,
        output_paths=output_paths,
    )


def print_summary(result: PipelineResult) -> None:
    best_scores = result.cv_results[result.model_name]
    print("ML raise decision summary")
    print("=" * 72)
    print(f"Model used : {result.model_name}")
    print(f"CV R2      : {best_scores['r2']:.4f}")
    print(f"CV RMSE    : {best_scores['rmse']:.4f}")
    print(f"Employees : {len(result.employees)}")
    print()
    print(result.decisions[["from_name", "email_count", "ml_score", "decision"]].to_string(index=False))
    print()
    print("Output files:")
    for label, path in result.output_paths.items():
        print(f"  {label}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EVA Group employee raise ML pipeline.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-emails", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation.")
    args = parser.parse_args()

    result = run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        min_emails=max(1, args.min_emails),
        save_plots=not args.no_plots,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
