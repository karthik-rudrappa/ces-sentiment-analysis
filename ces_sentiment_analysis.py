"""
============================================================
  Customer Effort Score — Sentiment Analysis Pipeline
============================================================
  Dual-layer analysis: VADER (primary) + TextBlob (cross-check)
  Produces 10 visualisation charts saved as PNG files.
  
  SETUP (run once in your VS Code terminal):
      pip install -r requirements.txt
      python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
  
  USAGE:
      Place data_ces.xlsx in the same folder as this script, then:
      python ces_sentiment_analysis.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import re
from datetime import datetime

# ── VADER
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ── TextBlob
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ============================================================
# 1.  CONFIGURATION
# ============================================================
DATA_FILE = "data_ces.xlsx"                 # <-- your Excel file
OUTPUT_FOLDER = "charts"                       # folder where PNGs are saved
STYLE = "seaborn-v0_8-whitegrid"       # matplotlib style

# Colour palette (corporate-friendly)
COLOURS = {
    "positive": "#2ecc71",
    "neutral": "#f39c12",
    "negative": "#e74c3c",
    "primary": "#2980b9",
    "secondary": "#8e44ad",
    "dark": "#2c3e50",
    "light": "#ecf0f1",
}

PALETTE_3 = [COLOURS["positive"], COLOURS["neutral"], COLOURS["negative"]]

# ============================================================
# 2.  HELPER — TEXT CLEANING
# ============================================================


def clean_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove non-alphanumeric
    (keep apostrophes for contractions)."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)          # collapse whitespace
    text = re.sub(r'[^\w\s\'\!\?\.\,]', '', text)  # keep basic punctuation
    return text.lower()

# ============================================================
# 3.  LOAD & PREPARE DATA
# ============================================================


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel("data_ces.xlsx")

    # ── standardise column names (strip whitespace, title-case)
    df.columns = df.columns.str.strip()

    # ── parse Percent Score  →  numeric float
    if "Percent Score" in df.columns:
        df["Percent Score"] = (
            df["Percent Score"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        df["Percent Score"] = pd.to_numeric(
            df["Percent Score"], errors="coerce")

    # ── parse Rating  →  numeric
    if "Rating" in df.columns:
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # ── parse dates
    for col in ["Result Date", "Date Picked"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # ── cleaned review text
    df["Review_Clean"] = df["Review"].apply(clean_text)

    # ── flag for rows that have actual review text
    df["Has_Review"] = df["Review_Clean"].apply(lambda x: len(x) > 0)

    print(
        f"\n✔ Loaded {len(df):,} rows  |  {df['Has_Review'].sum():,} with reviews  |  {(~df['Has_Review']).sum():,} empty\n")
    return df


# ============================================================
# 4.  SENTIMENT SCORING
# ============================================================
def score_vader(text: str, analyser: SentimentIntensityAnalyzer) -> dict:
    """Return VADER compound + pos/neu/neg."""
    if not text:
        return {"vader_compound": np.nan, "vader_pos": np.nan,
                "vader_neu": np.nan, "vader_neg": np.nan}
    scores = analyser.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos": scores["pos"],
        "vader_neu": scores["neu"],
        "vader_neg": scores["neg"],
    }


def score_textblob(text: str) -> dict:
    """Return TextBlob polarity & subjectivity."""
    if not text:
        return {"tb_polarity": np.nan, "tb_subjectivity": np.nan}
    blob = TextBlob(text)
    return {"tb_polarity": blob.sentiment.polarity,
            "tb_subjectivity": blob.sentiment.subjectivity}


def label_sentiment(compound: float) -> str:
    """Standard VADER thresholds."""
    if pd.isna(compound):
        return "No Review"
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def run_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    analyser = SentimentIntensityAnalyzer()

    vader_scores = df["Review_Clean"].apply(lambda t: score_vader(t, analyser))
    tb_scores = df["Review_Clean"].apply(score_textblob)

    vader_df = pd.DataFrame(vader_scores.tolist(), index=df.index)
    tb_df = pd.DataFrame(tb_scores.tolist(),    index=df.index)

    df = pd.concat([df, vader_df, tb_df], axis=1)
    df["Sentiment_Label"] = df["vader_compound"].apply(label_sentiment)

    # ── agreement flag  (both models agree on direction)
    def tb_label(pol):
        if pd.isna(pol):
            return "No Review"
        if pol > 0.0:
            return "Positive"
        elif pol < 0.0:
            return "Negative"
        else:
            return "Neutral"

    df["TB_Label"] = df["tb_polarity"].apply(tb_label)
    df["Agreement"] = df["Sentiment_Label"] == df["TB_Label"]

    print("✔ Sentiment scoring complete.\n")
    return df


# ============================================================
# 5.  RATING-BASED GROUND-TRUTH LABEL  (for validation)
# ============================================================
def rating_to_label(rating: float) -> str:
    """Map 1-7 CES rating → sentiment bucket."""
    if pd.isna(rating):
        return "No Rating"
    if rating >= 6:
        return "Positive"
    elif rating >= 4:
        return "Neutral"
    else:
        return "Negative"

# ============================================================
# 6.  VISUALISATIONS
# ============================================================


def setup_plot():
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_fig(name: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  💾 saved  →  {path}")


# ── 6.1  Overall Sentiment Distribution (Pie + Bar side by side)
def chart_01_overall_distribution(df: pd.DataFrame):
    setup_plot()
    counts = df["Sentiment_Label"].value_counts()
    # ensure order
    order = ["Positive", "Neutral", "Negative", "No Review"]
    counts = counts.reindex(
        [o for o in order if o in counts.index], fill_value=0)
    colours_ordered = [
        COLOURS.get(l.lower(), "#95a5a6") for l in counts.index
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Overall Sentiment Distribution",
                 fontsize=16, fontweight="bold", y=1.02)

    # ── pie
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colours_ordered, startangle=140,
        explode=[0.03]*len(counts), shadow=True
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[0].set_title("Proportion", fontsize=13)

    # ── bar
    bars = axes[1].bar(counts.index, counts.values, color=colours_ordered,
                       edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Counts", fontsize=13)
    axes[1].set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    save_fig("01_overall_sentiment_distribution.png")


# ── 6.2  Sentiment by Department  (stacked horizontal bar)
def chart_02_sentiment_by_department(df: pd.DataFrame):
    setup_plot()
    order = ["Positive", "Neutral", "Negative"]
    pivot = (
        df[df["Sentiment_Label"].isin(order)]
        .groupby(["Department", "Sentiment_Label"])
        .size()
        .unstack(fill_value=0)[order]
    )
    # sort departments by total volume descending
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.45)))
    fig.suptitle("Sentiment Breakdown by Department",
                 fontsize=16, fontweight="bold")

    pivot.plot(kind="barh", stacked=True, color=PALETTE_3,
               edgecolor="white", ax=ax, width=0.6)
    ax.set_xlabel("Number of Responses")
    ax.set_ylabel("")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")

    # ── percentage labels inside bars
    for i, dept in enumerate(pivot.index):
        total = pivot.loc[dept].sum()
        cumulative = 0
        for label, colour in zip(order, PALETTE_3):
            val = pivot.loc[dept, label]
            if val > 0 and (val/total) > 0.06:
                ax.text(cumulative + val/2, i,
                        f"{val/total*100:.0f}%", ha="center", va="center",
                        fontweight="bold", fontsize=9, color="white")
            cumulative += val

    plt.tight_layout()
    save_fig("02_sentiment_by_department.png")


# ── 6.3  Average VADER Compound Score by Department (dot plot)
def chart_03_avg_score_by_department(df: pd.DataFrame):
    setup_plot()
    reviewed = df[df["Has_Review"]]
    avg = reviewed.groupby("Department")["vader_compound"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(11, max(5, len(avg)*0.4)))
    fig.suptitle("Avg. VADER Compound Score by Department",
                 fontsize=16, fontweight="bold")

    colours_dot = [COLOURS["positive"] if v >= 0.05 else COLOURS["negative"]
                   if v <= -0.05 else COLOURS["neutral"] for v in avg]
    ax.barh(avg.index, avg.values, color=colours_dot,
            edgecolor="white", height=0.55)
    ax.axvline(0, color=COLOURS["dark"],
               linewidth=1, linestyle="--", alpha=0.6)

    for i, (dept, val) in enumerate(avg.items()):
        ax.text(val + (0.01 if val >= 0 else -0.01), i,
                f"{val:.3f}", va="center", ha="left" if val >= 0 else "right",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("Compound Score  (−1 → +1)")
    plt.tight_layout()
    save_fig("03_avg_vader_score_by_department.png")


# ── 6.4  Rating vs Predicted Sentiment  (heatmap / confusion matrix)
def chart_04_rating_vs_sentiment(df: pd.DataFrame):
    setup_plot()
    df_temp = df.copy()
    df_temp["Rating_Label"] = df_temp["Rating"].apply(rating_to_label)

    # only rows that have both
    mask = (df_temp["Sentiment_Label"] != "No Review") & (
        df_temp["Rating_Label"] != "No Rating")
    ct = pd.crosstab(df_temp.loc[mask, "Rating_Label"],
                     df_temp.loc[mask, "Sentiment_Label"])

    order = ["Positive", "Neutral", "Negative"]
    ct = ct.reindex(index=order, columns=order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Rating-Based Label  vs  Predicted Sentiment\n(Validation Heatmap)",
                 fontsize=15, fontweight="bold")

    sns.heatmap(ct, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
                linewidths=1, linecolor="white", cbar_kws={"label": "Count"},
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted (VADER)", fontsize=12)
    ax.set_ylabel("Actual (Rating)", fontsize=12)
    plt.tight_layout()
    save_fig("04_rating_vs_predicted_heatmap.png")


# ── 6.5  Sentiment Trend Over Time  (line chart)
def chart_05_sentiment_trend(df: pd.DataFrame):
    setup_plot()
    df_temp = df[df["Result Date"].notna()].copy()
    df_temp["Week"] = df_temp["Result Date"].dt.to_period("W").dt.start_time

    trend = (
        df_temp[df_temp["Sentiment_Label"].isin(
            ["Positive", "Neutral", "Negative"])]
        .groupby(["Week", "Sentiment_Label"])
        .size()
        .unstack(fill_value=0)[["Positive", "Neutral", "Negative"]]
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Sentiment Trend Over Time (Weekly)",
                 fontsize=16, fontweight="bold")

    for label, colour in zip(["Positive", "Neutral", "Negative"], PALETTE_3):
        ax.plot(trend.index, trend[label], marker="o", color=colour,
                linewidth=2.2, markersize=5, label=label)
        ax.fill_between(trend.index, trend[label], alpha=0.08, color=colour)

    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    ax.xaxis.set_tick_params(rotation=35)
    plt.tight_layout()
    save_fig("05_sentiment_trend_over_time.png")


# ── 6.6  VADER vs TextBlob Polarity Scatter  (agreement map)
def chart_06_vader_vs_textblob(df: pd.DataFrame):
    setup_plot()
    reviewed = df[df["Has_Review"]].dropna(
        subset=["vader_compound", "tb_polarity"])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("VADER Compound  vs  TextBlob Polarity\n(Model Agreement Map)",
                 fontsize=15, fontweight="bold")

    colour_map = {"Positive": COLOURS["positive"],
                  "Neutral": COLOURS["neutral"],
                  "Negative": COLOURS["negative"]}

    for label in ["Positive", "Neutral", "Negative"]:
        subset = reviewed[reviewed["Sentiment_Label"] == label]
        ax.scatter(subset["vader_compound"], subset["tb_polarity"],
                   c=colour_map[label], label=label, s=55, edgecolors="white",
                   linewidth=0.8, alpha=0.85)

    # reference lines
    ax.axhline(0, color=COLOURS["dark"],
               linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color=COLOURS["dark"],
               linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.05,  color=COLOURS["dark"],
               linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axvline(-0.05, color=COLOURS["dark"],
               linewidth=0.5, linestyle=":", alpha=0.4)

    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("TextBlob Polarity")
    ax.legend(title="VADER Label", fontsize=10)

    # agreement % annotation
    agree_pct = reviewed["Agreement"].mean() * 100
    ax.text(0.02, 0.97, f"Model agreement: {agree_pct:.1f}%",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOURS["light"], edgecolor="grey"))

    plt.tight_layout()
    save_fig("06_vader_vs_textblob_scatter.png")


# ── 6.7  Top Positive & Negative Reviews  (table-style figure)
def chart_07_top_reviews(df: pd.DataFrame, n: int = 5):
    setup_plot()
    reviewed = df[df["Has_Review"]].copy()

    top_pos = reviewed.nlargest(n, "vader_compound")[
        ["Review", "vader_compound", "Department", "Rating"]]
    top_neg = reviewed.nsmallest(n, "vader_compound")[
        ["Review", "vader_compound", "Department", "Rating"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, n*1.1)))
    fig.suptitle(f"Top {n} Positive & Negative Reviews (by VADER Score)",
                 fontsize=16, fontweight="bold")

    for ax, data, title, colour in [
        (axes[0], top_pos, "Most Positive", COLOURS["positive"]),
        (axes[1], top_neg, "Most Negative", COLOURS["negative"])
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold",
                     color=colour, pad=12)

        y = 0.95
        for idx, (_, row) in enumerate(data.iterrows()):
            review_text = str(row["Review"])
            # wrap long reviews
            if len(review_text) > 90:
                review_text = review_text[:87] + "..."
            score_text = f"Score: {row['vader_compound']:.3f}  |  Dept: {row['Department']}  |  Rating: {int(row['Rating']) if pd.notna(row['Rating']) else 'N/A'}"
            ax.text(0.02, y, f"{idx+1}. {review_text}",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    verticalalignment="top", wrap=True)
            ax.text(0.02, y - 0.045, score_text,
                    transform=ax.transAxes, fontsize=9, color="grey",
                    verticalalignment="top")
            y -= 0.19

    plt.tight_layout()
    save_fig("07_top_positive_negative_reviews.png")


# ── 6.8  Sentiment Distribution by Advisor (top 10 by volume)
def chart_08_sentiment_by_advisor(df: pd.DataFrame):
    setup_plot()
    order = ["Positive", "Neutral", "Negative"]
    # filter out N/A advisors
    df_adv = df[(df["Advisor"].notna()) & (df["Advisor"].str.strip()
                                           != "N/A") & (df["Sentiment_Label"].isin(order))]

    # top 10 advisors by total responses
    top10 = df_adv["Advisor"].value_counts().head(10).index.tolist()
    df_adv = df_adv[df_adv["Advisor"].isin(top10)]

    pivot = (
        df_adv.groupby(["Advisor", "Sentiment_Label"])
        .size().unstack(fill_value=0)[order]
    )
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.5)))
    fig.suptitle("Sentiment by Advisor  (Top 10 by Volume)",
                 fontsize=16, fontweight="bold")

    pivot.plot(kind="barh", stacked=True, color=PALETTE_3,
               edgecolor="white", ax=ax, width=0.6)
    ax.set_xlabel("Responses")
    ax.set_ylabel("")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    save_fig("08_sentiment_by_advisor.png")


# ── 6.9  Rating Distribution  (violin / box combo)
def chart_09_rating_distribution(df: pd.DataFrame):
    setup_plot()
    order = ["Positive", "Neutral", "Negative"]
    df_plot = df[df["Sentiment_Label"].isin(order)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("CES Rating Distribution by Sentiment Label",
                 fontsize=16, fontweight="bold")

    sns.boxplot(data=df_plot, x="Sentiment_Label", y="Rating",
                order=order, palette=dict(zip(order, PALETTE_3)),
                width=0.45, linewidth=1.8, ax=ax,
                flierprops=dict(marker="o", markerfacecolor="grey", markersize=5))

    # overlay individual points (jitter)
    sns.stripplot(data=df_plot, x="Sentiment_Label", y="Rating",
                  order=order, color=COLOURS["dark"],
                  alpha=0.35, size=4, jitter=0.12, ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("CES Rating (1–7)")
    ax.set_ylim(0, 8)
    plt.tight_layout()
    save_fig("09_rating_distribution_by_sentiment.png")


# ── 6.10  Subjectivity vs Polarity  (TextBlob)  coloured by department
def chart_10_subjectivity_map(df: pd.DataFrame):
    setup_plot()
    reviewed = df[df["Has_Review"]].dropna(
        subset=["tb_polarity", "tb_subjectivity"])

    departments = reviewed["Department"].unique()
    cmap = plt.cm.get_cmap("tab10", len(departments))

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Subjectivity vs Polarity by Department\n(TextBlob)",
                 fontsize=15, fontweight="bold")

    for i, dept in enumerate(departments):
        subset = reviewed[reviewed["Department"] == dept]
        ax.scatter(subset["tb_polarity"], subset["tb_subjectivity"],
                   c=[cmap(i)], label=dept, s=55, edgecolors="white",
                   linewidth=0.7, alpha=0.8)

    ax.axhline(0.5, color=COLOURS["dark"],
               linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axvline(0,   color=COLOURS["dark"],
               linewidth=0.8, linestyle="--", alpha=0.45)

    ax.set_xlabel("Polarity  (−1 Negative → +1 Positive)")
    ax.set_ylabel("Subjectivity  (0 Objective → 1 Subjective)")

    ax.legend(title="Department", bbox_to_anchor=(
        1.02, 1), loc="upper left", fontsize=9)

    # quadrant labels
    ax.text(-0.95, 0.95, "Objective\nNegative",
            fontsize=8, color="grey", alpha=0.7, va="top")
    ax.text(0.75, 0.95, "Objective\nPositive",
            fontsize=8, color="grey", alpha=0.7, va="top")
    ax.text(-0.95, 0.05, "Subjective\nNegative", fontsize=8,
            color="grey", alpha=0.7, va="bottom")
    ax.text(0.75, 0.05, "Subjective\nPositive",  fontsize=8,
            color="grey", alpha=0.7, va="bottom")

    plt.tight_layout()
    save_fig("10_subjectivity_vs_polarity.png")


# ============================================================
# 7.  SUMMARY REPORT  (printed + saved as CSV)
# ============================================================
def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  SENTIMENT ANALYSIS SUMMARY")
    print("="*60)

    total = len(df)
    with_review = df["Has_Review"].sum()
    no_review = total - with_review

    print(f"\n  Total responses          : {total:,}")
    print(f"  With review text         : {with_review:,}")
    print(f"  Without review text      : {no_review:,}")

    labelled = df[df["Sentiment_Label"] != "No Review"]
    print(f"\n  — Sentiment Breakdown (reviewed only) —")
    for label in ["Positive", "Neutral", "Negative"]:
        count = (labelled["Sentiment_Label"] == label).sum()
        pct = count / len(labelled) * 100
        print(f"    {label:12s} : {count:5,}  ({pct:5.1f}%)")

    agree = labelled["Agreement"].mean() * 100
    print(f"\n  VADER ↔ TextBlob agreement : {agree:.1f}%")

    avg_compound = labelled["vader_compound"].mean()
    print(f"  Overall avg compound score : {avg_compound:+.3f}")

    print(f"\n  — Avg Compound by Department —")
    dept_avg = (
        labelled.groupby("Department")["vader_compound"]
        .agg(["mean", "count"])
        .sort_values("mean", ascending=False)
    )
    for dept, row in dept_avg.iterrows():
        print(f"    {dept:40s} {row['mean']:+.3f}   (n={int(row['count'])})")

    print("\n" + "="*60 + "\n")


def export_results(df: pd.DataFrame):
    """Save enriched dataframe with sentiment columns."""
    export_cols = [
        "Result Date", "C Number", "Department", "Advisor", "Rating",
        "Review", "Percent Score",
        "vader_compound", "vader_pos", "vader_neu", "vader_neg",
        "Sentiment_Label",
        "tb_polarity", "tb_subjectivity", "TB_Label", "Agreement"
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    out_path = "ces_sentiment_results.xlsx"
    df[export_cols].to_excel(out_path, index=False, engine="openpyxl")
    print(f"  💾 enriched results saved → {out_path}\n")


# ============================================================
# 8.  MAIN
# ============================================================
def main():
    print("\n🔄 Loading data …")
    df = load_data(DATA_FILE)

    print("🔄 Running sentiment analysis …")
    df = run_sentiment(df)

    print("🔄 Generating charts …\n")
    chart_01_overall_distribution(df)
    chart_02_sentiment_by_department(df)
    chart_03_avg_score_by_department(df)
    chart_04_rating_vs_sentiment(df)
    chart_05_sentiment_trend(df)
    chart_06_vader_vs_textblob(df)
    chart_07_top_reviews(df)
    chart_08_sentiment_by_advisor(df)
    chart_09_rating_distribution(df)
    chart_10_subjectivity_map(df)

    print("\n🔄 Summary …")
    print_summary(df)

    print("🔄 Exporting enriched data …")
    export_results(df)

    print("✅ All done! Charts saved in  ./{}/  \n".format(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
