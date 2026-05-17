from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TIKTOK_DIR = ROOT_DIR / "tiktok scraper"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "reports" / "tiktok_comparison"
DEFAULT_CSV_A = DEFAULT_TIKTOK_DIR / "maluukss0_tiktok_20260508_213407.csv"
DEFAULT_CSV_B = DEFAULT_TIKTOK_DIR / "samaalwagih__tiktok_20260508_213457.csv"

REQUIRED_COLUMNS = {"views", "likes", "created_at"}


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"TikTok CSV not found: {path}")

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    frame = frame.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    frame["like_rate"] = frame["likes"].div(frame["views"].replace(0, pd.NA))
    return frame


def summarize(frame: pd.DataFrame, label: str) -> pd.Series:
    return pd.Series(
        {
            "posts": len(frame),
            "views_total": frame["views"].sum(),
            "views_avg": frame["views"].mean(),
            "views_median": frame["views"].median(),
            "likes_total": frame["likes"].sum(),
            "likes_avg": frame["likes"].mean(),
            "likes_median": frame["likes"].median(),
            "like_rate_avg": frame["like_rate"].mean(),
            "date_min": frame["created_at"].min(),
            "date_max": frame["created_at"].max(),
        },
        name=label,
    )


def top_posts_by_views(frame: pd.DataFrame, limit: int = 5) -> pd.DataFrame:
    return frame.nlargest(limit, "views")[["created_at", "views", "likes", "like_rate"]]


def format_compact(value: float, _position: object = None) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        compact = f"{value / 1_000_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}B"
    if absolute >= 1_000_000:
        compact = f"{value / 1_000_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}M"
    if absolute >= 1_000:
        compact = f"{value / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{compact}K"
    return f"{value:,.0f}"


def group_by_week(frame: pd.DataFrame) -> pd.DataFrame:
    dated = frame.dropna(subset=["created_at"]).copy()
    dated["week_start"] = dated["created_at"].dt.to_period("W").dt.start_time
    return dated.groupby("week_start")[["views", "likes"]].sum().sort_index()


def save_totals_plot(output_dir: Path, summary: pd.DataFrame) -> Path:
    totals = summary[["views_total", "likes_total"]].copy()
    ax = totals.plot(kind="bar", figsize=(8, 4))
    ax.yaxis.set_major_formatter(FuncFormatter(format_compact))
    ax.set_title("Total views and likes")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()

    path = output_dir / "tiktok_totals.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def save_weekly_trend_plot(
    output_dir: Path,
    frame_a: pd.DataFrame,
    label_a: str,
    frame_b: pd.DataFrame,
    label_b: str,
) -> Path:
    trend_a = group_by_week(frame_a)
    trend_b = group_by_week(frame_b)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    trend_a["views"].plot(ax=axes[0], label=label_a)
    trend_b["views"].plot(ax=axes[0], label=label_b)
    axes[0].set_title("Views per week")
    axes[0].set_ylabel("Views")
    axes[0].yaxis.set_major_formatter(FuncFormatter(format_compact))
    axes[0].legend()

    trend_a["likes"].plot(ax=axes[1], label=label_a)
    trend_b["likes"].plot(ax=axes[1], label=label_b)
    axes[1].set_title("Likes per week")
    axes[1].set_ylabel("Likes")
    axes[1].yaxis.set_major_formatter(FuncFormatter(format_compact))
    axes[1].legend()

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    path = output_dir / "tiktok_weekly_trends.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def compare_accounts(
    csv_a: Path = DEFAULT_CSV_A,
    csv_b: Path = DEFAULT_CSV_B,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    top_n: int = 5,
    save_plots: bool = True,
) -> dict[str, Path | pd.DataFrame]:
    frame_a = load_csv(csv_a)
    frame_b = load_csv(csv_b)
    label_a = csv_a.stem
    label_b = csv_b.stem

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat([summarize(frame_a, label_a), summarize(frame_b, label_b)], axis=1).T
    top_a = top_posts_by_views(frame_a, limit=top_n)
    top_b = top_posts_by_views(frame_b, limit=top_n)

    summary_path = output_dir / "summary.csv"
    top_a_path = output_dir / f"{label_a}_top_posts.csv"
    top_b_path = output_dir / f"{label_b}_top_posts.csv"
    summary.to_csv(summary_path)
    top_a.to_csv(top_a_path, index=False)
    top_b.to_csv(top_b_path, index=False)

    payload: dict[str, Path | pd.DataFrame] = {
        "summary": summary,
        "top_a": top_a,
        "top_b": top_b,
        "summary_path": summary_path,
        "top_a_path": top_a_path,
        "top_b_path": top_b_path,
    }

    if save_plots:
        payload["totals_plot"] = save_totals_plot(output_dir, summary)
        payload["weekly_trends_plot"] = save_weekly_trend_plot(output_dir, frame_a, label_a, frame_b, label_b)

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two TikTok scraper CSV exports.")
    parser.add_argument("--csv-a", type=Path, default=DEFAULT_CSV_A)
    parser.add_argument("--csv-b", type=Path, default=DEFAULT_CSV_B)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation.")
    args = parser.parse_args()

    result = compare_accounts(
        csv_a=args.csv_a,
        csv_b=args.csv_b,
        output_dir=args.output_dir,
        top_n=max(1, args.top_n),
        save_plots=not args.no_plots,
    )

    print("TikTok comparison summary")
    print("=" * 72)
    print(result["summary"].to_string())
    print()
    print("Output files:")
    for key, value in result.items():
        if key.endswith("_path") or key.endswith("_plot"):
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
