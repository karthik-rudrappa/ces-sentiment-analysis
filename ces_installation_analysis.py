"""
============================================================
  INSTALLATION-FOCUSED SENTIMENT ANALYSIS
============================================================
  Deep-dive on installation reviews only.
  Full VADER + TextBlob dual-layer sentiment analysis.
  
  Filters:
    - Department = "Install" OR
    - Review mentions installation-related keywords
  
  Charts produced (10 total):
    01. Overall sentiment distribution (Install only)
    02. Sentiment by contractor/install team
    03. Average VADER score by contractor
    04. Rating vs Predicted Sentiment (validation heatmap)
    05. Installation sentiment trend over time
    06. VADER vs TextBlob scatter (agreement map)
    07. Top positive & negative installation reviews
    08. Rating distribution by sentiment
    09. Subjectivity vs Polarity (TextBlob)
    10. Installation failure keywords by rating band
  
  SETUP (run once):
      pip install -r requirements.txt
      python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
  
  USAGE:
      python ces_installation_analysis.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import re
from collections import Counter
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import stopwords

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_FILE       = "data_ces.xlsx"
OUTPUT_FOLDER   = "charts_installation"
STYLE           = "seaborn-v0_8-whitegrid"

# Installation-specific keywords for filtering reviews
INSTALL_KEYWORDS = [
    "install", "installation", "installed", "installer", "installers",
    "engineer", "engineers", "technician", "tech", "contractor",
    "cable", "cables", "cabling", "wire", "wiring", "fibre", "fiber",
    "dig", "digging", "dug", "trench", "trenching",
    "appointment", "booked", "booking", "scheduled", "schedule",
    "visit", "visited", "came", "arrived", "turn up", "turned up",
    "setup", "set up", "connect", "connection", "connected"
]

# Colour palette
COLOURS = {
    "positive" : "#2ecc71",
    "neutral"  : "#f39c12",
    "negative" : "#e74c3c",
    "primary"  : "#2980b9",
    "secondary": "#8e44ad",
    "dark"     : "#2c3e50",
    "light"    : "#ecf0f1",
}

PALETTE_3 = [COLOURS["positive"], COLOURS["neutral"], COLOURS["negative"]]

# ============================================================
# 2. HELPERS
# ============================================================
def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\'\!\?\.\,]', '', text)
    return text.lower()

def is_installation_review(row: pd.Series) -> bool:
    """
    Returns True if this review is installation-related.
    Criteria: Department is "Install" OR review text mentions install keywords.
    """
    # check department
    if pd.notna(row.get("Department")):
        dept = str(row["Department"]).strip().lower()
        if "install" in dept:
            return True
    
    # check review text for install keywords
    review_text = row.get("Review_Clean", "")
    if review_text:
        for kw in INSTALL_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', review_text):
                return True
    
    return False

def setup_plot():
    plt.style.use(STYLE)
    plt.rcParams.update({
        "font.family"  : "sans-serif",
        "font.size"    : 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor" : "white",
        "axes.spines.top"   : False,
        "axes.spines.right" : False,
    })

def save_fig(name: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  💾 saved  →  {path}")

# ============================================================
# 3. LOAD & FILTER
# ============================================================
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()

    # parse Percent Score
    if "Percent Score" in df.columns:
        df["Percent Score"] = (
            df["Percent Score"].astype(str)
            .str.replace("%", "", regex=False).str.strip()
        )
        df["Percent Score"] = pd.to_numeric(df["Percent Score"], errors="coerce")

    # parse Rating
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # parse dates
    for col in ["Result Date", "Date Picked"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # cleaned review text
    df["Review_Clean"] = df["Review"].apply(clean_text)
    df["Has_Review"]   = df["Review_Clean"].apply(lambda x: len(x) > 0)

    print(f"\n✔ Loaded {len(df):,} total rows from dataset")

    # FILTER: installation reviews only
    df["Is_Installation"] = df.apply(is_installation_review, axis=1)
    df_install = df[df["Is_Installation"]].copy()

    print(f"✔ Filtered to {len(df_install):,} installation-related reviews")
    print(f"  - {df_install['Has_Review'].sum():,} with review text")
    print(f"  - {(~df_install['Has_Review']).sum():,} without review text\n")

    return df_install

# ============================================================
# 4. SENTIMENT SCORING
# ============================================================
def score_vader(text: str, analyser: SentimentIntensityAnalyzer) -> dict:
    if not text:
        return {"vader_compound": np.nan, "vader_pos": np.nan,
                "vader_neu": np.nan, "vader_neg": np.nan}
    scores = analyser.polarity_scores(text)
    return {
        "vader_compound": scores["compound"],
        "vader_pos"     : scores["pos"],
        "vader_neu"     : scores["neu"],
        "vader_neg"     : scores["neg"],
    }

def score_textblob(text: str) -> dict:
    if not text:
        return {"tb_polarity": np.nan, "tb_subjectivity": np.nan}
    blob = TextBlob(text)
    return {"tb_polarity": blob.sentiment.polarity,
            "tb_subjectivity": blob.sentiment.subjectivity}

def label_sentiment(compound: float) -> str:
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
    tb_scores    = df["Review_Clean"].apply(score_textblob)

    vader_df = pd.DataFrame(vader_scores.tolist(), index=df.index)
    tb_df    = pd.DataFrame(tb_scores.tolist(),    index=df.index)

    df = pd.concat([df, vader_df, tb_df], axis=1)
    df["Sentiment_Label"] = df["vader_compound"].apply(label_sentiment)

    # agreement flag
    def tb_label(pol):
        if pd.isna(pol): return "No Review"
        if pol > 0.0:  return "Positive"
        elif pol < 0.0: return "Negative"
        else:           return "Neutral"

    df["TB_Label"]  = df["tb_polarity"].apply(tb_label)
    df["Agreement"] = df["Sentiment_Label"] == df["TB_Label"]

    print("✔ Sentiment scoring complete.\n")
    return df

# ============================================================
# 5. RATING-BASED GROUND-TRUTH LABEL
# ============================================================
def rating_to_label(rating: float) -> str:
    if pd.isna(rating):  return "No Rating"
    if rating >= 6:      return "Positive"
    elif rating >= 4:    return "Neutral"
    else:                return "Negative"

# ============================================================
# 6. VISUALISATIONS
# ============================================================

# ── 6.1  Overall Sentiment Distribution (Pie + Bar)
def chart_01_overall_distribution(df: pd.DataFrame):
    setup_plot()
    counts = df["Sentiment_Label"].value_counts()
    order  = ["Positive", "Neutral", "Negative", "No Review"]
    counts = counts.reindex([o for o in order if o in counts.index], fill_value=0)
    colours_ordered = [COLOURS.get(l.lower(), "#95a5a6") for l in counts.index]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Installation Sentiment Distribution", fontsize=16, fontweight="bold", y=1.02)

    # pie
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colours_ordered, startangle=140,
        explode=[0.03]*len(counts), shadow=True
    )
    for at in autotexts:
        at.set_fontweight("bold")
    axes[0].set_title("Proportion", fontsize=13)

    # bar
    bars = axes[1].bar(counts.index, counts.values, color=colours_ordered, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=12)
    axes[1].set_ylabel("Count")
    axes[1].set_title("Counts", fontsize=13)
    axes[1].set_ylim(0, counts.max() * 1.15)

    plt.tight_layout()
    save_fig("01_install_sentiment_distribution.png")


# ── 6.2  Sentiment by Contractor/Install Team
def chart_02_sentiment_by_contractor(df: pd.DataFrame):
    setup_plot()
    order = ["Positive", "Neutral", "Negative"]
    
    # Use Advisor as proxy for contractor/install team
    df_adv = df[(df["Advisor"].notna()) & (df["Advisor"].str.strip().str.upper() != "N/A")].copy()
    df_adv = df_adv[df_adv["Sentiment_Label"].isin(order)]
    
    if len(df_adv) == 0:
        print("  ⚠ No contractor/advisor data — skipping chart 02.")
        return
    
    pivot = (
        df_adv.groupby(["Advisor", "Sentiment_Label"])
        .size().unstack(fill_value=0)[order]
    )
    # top 10 by volume
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).head(10).drop(columns="Total")
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot)*0.5)))
    fig.suptitle("Installation Sentiment by Contractor/Team (Top 10)", fontsize=16, fontweight="bold")

    pivot.plot(kind="barh", stacked=True, color=PALETTE_3, edgecolor="white", ax=ax, width=0.6)
    ax.set_xlabel("Number of Responses")
    ax.set_ylabel("")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")

    # percentage labels
    for i, team in enumerate(pivot.index):
        total = pivot.loc[team].sum()
        cum = 0
        for label, colour in zip(order, PALETTE_3):
            val = pivot.loc[team, label]
            if val > 0 and (val/total) > 0.06:
                ax.text(cum + val/2, i,
                        f"{val/total*100:.0f}%", ha="center", va="center",
                        fontweight="bold", fontsize=9, color="white")
            cum += val

    plt.tight_layout()
    save_fig("02_install_sentiment_by_contractor.png")


# ── 6.3  Average VADER Score by Contractor
def chart_03_avg_score_by_contractor(df: pd.DataFrame):
    setup_plot()
    df_adv = df[(df["Advisor"].notna()) & (df["Advisor"].str.strip().str.upper() != "N/A") & df["Has_Review"]].copy()
    
    if len(df_adv) == 0:
        print("  ⚠ No contractor data with reviews — skipping chart 03.")
        return
    
    avg = df_adv.groupby("Advisor")["vader_compound"].agg(["mean", "count"])
    avg = avg[avg["count"] >= 3].sort_values("mean", ascending=True).head(12)

    fig, ax = plt.subplots(figsize=(11, max(5, len(avg)*0.4)))
    fig.suptitle("Avg. Installation VADER Score by Contractor (min 3 reviews)", fontsize=16, fontweight="bold")

    colours_dot = [COLOURS["positive"] if v >= 0.05 else COLOURS["negative"] if v <= -0.05 else COLOURS["neutral"] for v in avg["mean"]]
    ax.barh(avg.index, avg["mean"], color=colours_dot, edgecolor="white", height=0.55)
    ax.axvline(0, color=COLOURS["dark"], linewidth=1, linestyle="--", alpha=0.6)

    for i, (contractor, row) in enumerate(avg.iterrows()):
        ax.text(row["mean"] + (0.01 if row["mean"] >= 0 else -0.01), i,
                f"{row['mean']:.3f} (n={int(row['count'])})", va="center",
                ha="left" if row["mean"] >= 0 else "right",
                fontsize=9, fontweight="bold")

    ax.set_xlabel("Compound Score  (−1 → +1)")
    plt.tight_layout()
    save_fig("03_install_avg_vader_by_contractor.png")


# ── 6.4  Rating vs Predicted Sentiment (heatmap / confusion)
def chart_04_rating_vs_sentiment(df: pd.DataFrame):
    setup_plot()
    df_temp = df.copy()
    df_temp["Rating_Label"] = df_temp["Rating"].apply(rating_to_label)

    mask = (df_temp["Sentiment_Label"] != "No Review") & (df_temp["Rating_Label"] != "No Rating")
    ct = pd.crosstab(df_temp.loc[mask, "Rating_Label"],
                     df_temp.loc[mask, "Sentiment_Label"])

    order = ["Positive", "Neutral", "Negative"]
    ct = ct.reindex(index=order, columns=order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Installation: Rating-Based Label vs Predicted Sentiment\n(Validation Heatmap)",
                 fontsize=15, fontweight="bold")

    sns.heatmap(ct, annot=True, fmt="d", cmap="RdYlGn", ax=ax,
                linewidths=1, linecolor="white", cbar_kws={"label": "Count"},
                annot_kws={"size": 16, "weight": "bold"})
    ax.set_xlabel("Predicted (VADER)", fontsize=12)
    ax.set_ylabel("Actual (Rating)", fontsize=12)
    plt.tight_layout()
    save_fig("04_install_rating_vs_predicted.png")


# ── 6.5  Installation Sentiment Trend Over Time
def chart_05_sentiment_trend(df: pd.DataFrame):
    setup_plot()
    df_temp = df[df["Result Date"].notna()].copy()
    df_temp["Week"] = df_temp["Result Date"].dt.to_period("W").dt.start_time

    trend = (
        df_temp[df_temp["Sentiment_Label"].isin(["Positive","Neutral","Negative"])]
        .groupby(["Week","Sentiment_Label"])
        .size()
        .unstack(fill_value=0)[["Positive","Neutral","Negative"]]
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle("Installation Sentiment Trend Over Time (Weekly)", fontsize=16, fontweight="bold")

    for label, colour in zip(["Positive","Neutral","Negative"], PALETTE_3):
        ax.plot(trend.index, trend[label], marker="o", color=colour,
                linewidth=2.2, markersize=5, label=label)
        ax.fill_between(trend.index, trend[label], alpha=0.08, color=colour)

    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment")
    ax.xaxis.set_tick_params(rotation=35)
    plt.tight_layout()
    save_fig("05_install_sentiment_trend.png")


# ── 6.6  VADER vs TextBlob Scatter (agreement map)
def chart_06_vader_vs_textblob(df: pd.DataFrame):
    setup_plot()
    reviewed = df[df["Has_Review"]].dropna(subset=["vader_compound","tb_polarity"])

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle("Installation: VADER vs TextBlob Polarity\n(Model Agreement Map)",
                 fontsize=15, fontweight="bold")

    colour_map = {"Positive": COLOURS["positive"],
                  "Neutral" : COLOURS["neutral"],
                  "Negative": COLOURS["negative"]}

    for label in ["Positive","Neutral","Negative"]:
        subset = reviewed[reviewed["Sentiment_Label"] == label]
        ax.scatter(subset["vader_compound"], subset["tb_polarity"],
                   c=colour_map[label], label=label, s=55, edgecolors="white",
                   linewidth=0.8, alpha=0.85)

    ax.axhline(0, color=COLOURS["dark"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color=COLOURS["dark"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0.05,  color=COLOURS["dark"], linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axvline(-0.05, color=COLOURS["dark"], linewidth=0.5, linestyle=":", alpha=0.4)

    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("TextBlob Polarity")
    ax.legend(title="VADER Label", fontsize=10)

    agree_pct = reviewed["Agreement"].mean() * 100
    ax.text(0.02, 0.97, f"Model agreement: {agree_pct:.1f}%",
            transform=ax.transAxes, fontsize=11, fontweight="bold",
            va="top", bbox=dict(boxstyle="round,pad=0.3", facecolor=COLOURS["light"], edgecolor="grey"))

    plt.tight_layout()
    save_fig("06_install_vader_vs_textblob.png")


# ── 6.7  Top Positive & Negative Installation Reviews
def chart_07_top_reviews(df: pd.DataFrame, n: int = 5):
    setup_plot()
    reviewed = df[df["Has_Review"]].copy()

    top_pos = reviewed.nlargest(n, "vader_compound")[["Review","vader_compound","Advisor","Rating"]]
    top_neg = reviewed.nsmallest(n, "vader_compound")[["Review","vader_compound","Advisor","Rating"]]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(5, n*1.1)))
    fig.suptitle(f"Top {n} Installation Reviews (by VADER Score)",
                 fontsize=16, fontweight="bold")

    for ax, data, title, colour in [
        (axes[0], top_pos, "Most Positive", COLOURS["positive"]),
        (axes[1], top_neg, "Most Negative", COLOURS["negative"])
    ]:
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", color=colour, pad=12)

        y = 0.95
        for idx, (_, row) in enumerate(data.iterrows()):
            review_text = str(row["Review"])
            if len(review_text) > 90:
                review_text = review_text[:87] + "..."
            score_text = f"Score: {row['vader_compound']:.3f}  |  Advisor: {row['Advisor']}  |  Rating: {int(row['Rating']) if pd.notna(row['Rating']) else 'N/A'}"
            ax.text(0.02, y, f"{idx+1}. {review_text}",
                    transform=ax.transAxes, fontsize=10, fontweight="bold",
                    verticalalignment="top", wrap=True)
            ax.text(0.02, y - 0.045, score_text,
                    transform=ax.transAxes, fontsize=9, color="grey",
                    verticalalignment="top")
            y -= 0.19

    plt.tight_layout()
    save_fig("07_install_top_reviews.png")


# ── 6.8  Rating Distribution by Sentiment
def chart_08_rating_distribution(df: pd.DataFrame):
    setup_plot()
    order = ["Positive", "Neutral", "Negative"]
    df_plot = df[df["Sentiment_Label"].isin(order)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Installation CES Rating Distribution by Sentiment", fontsize=16, fontweight="bold")

    sns.boxplot(data=df_plot, x="Sentiment_Label", y="Rating",
                order=order, palette=dict(zip(order, PALETTE_3)),
                width=0.45, linewidth=1.8, ax=ax,
                flierprops=dict(marker="o", markerfacecolor="grey", markersize=5))

    sns.stripplot(data=df_plot, x="Sentiment_Label", y="Rating",
                  order=order, color=COLOURS["dark"],
                  alpha=0.35, size=4, jitter=0.12, ax=ax)

    ax.set_xlabel("")
    ax.set_ylabel("CES Rating (1–7)")
    ax.set_ylim(0, 8)
    plt.tight_layout()
    save_fig("08_install_rating_distribution.png")


# ── 6.9  Subjectivity vs Polarity (TextBlob)
def chart_09_subjectivity_map(df: pd.DataFrame):
    setup_plot()
    reviewed = df[df["Has_Review"]].dropna(subset=["tb_polarity","tb_subjectivity"])

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Installation: Subjectivity vs Polarity\n(TextBlob)",
                 fontsize=15, fontweight="bold")

    colour_map = {"Positive": COLOURS["positive"],
                  "Neutral" : COLOURS["neutral"],
                  "Negative": COLOURS["negative"]}

    for label in ["Positive", "Neutral", "Negative"]:
        subset = reviewed[reviewed["Sentiment_Label"] == label]
        ax.scatter(subset["tb_polarity"], subset["tb_subjectivity"],
                   c=colour_map[label], label=label, s=55, edgecolors="white",
                   linewidth=0.7, alpha=0.8)

    ax.axhline(0.5, color=COLOURS["dark"], linewidth=0.8, linestyle="--", alpha=0.45)
    ax.axvline(0,   color=COLOURS["dark"], linewidth=0.8, linestyle="--", alpha=0.45)

    ax.set_xlabel("Polarity  (−1 Negative → +1 Positive)")
    ax.set_ylabel("Subjectivity  (0 Objective → 1 Subjective)")
    ax.legend(title="Sentiment")

    # quadrant labels
    ax.text(-0.95, 0.95, "Objective\nNegative", fontsize=8, color="grey", alpha=0.7, va="top")
    ax.text( 0.75, 0.95, "Objective\nPositive",  fontsize=8, color="grey", alpha=0.7, va="top")
    ax.text(-0.95, 0.05, "Subjective\nNegative", fontsize=8, color="grey", alpha=0.7, va="bottom")
    ax.text( 0.75, 0.05, "Subjective\nPositive",  fontsize=8, color="grey", alpha=0.7, va="bottom")

    plt.tight_layout()
    save_fig("09_install_subjectivity_vs_polarity.png")


# ── 6.10  Installation Failure Keywords by Rating Band
def chart_10_failure_keywords(df: pd.DataFrame):
    setup_plot()
    neg_df = df[(df["Sentiment_Label"] == "Negative") & df["Has_Review"]].copy()

    # rating bands
    def get_band(r):
        if pd.isna(r): return "Unknown"
        if r >= 6: return "Promoter (6-7)"
        if r >= 4: return "Passive (4-5)"
        return "Detractor (1-3)"

    neg_df["Band"] = neg_df["Rating"].apply(get_band)

    bands = ["Detractor (1-3)", "Passive (4-5)"]
    fig, axes = plt.subplots(1, len(bands), figsize=(15, 5.5), sharey=False)
    fig.patch.set_facecolor("white")
    fig.suptitle("Top Installation Failure Keywords by Rating Band", fontsize=16, fontweight="bold")

    stop = set(stopwords.words("english")) | {"", "like", "also", "one", "get", "got", "still", "really", "much", "even", "quite", "already"}

    for ax, band in zip(axes, bands):
        subset = neg_df[neg_df["Band"] == band]
        words  = []
        for txt in subset["Review_Clean"]:
            words.extend(re.findall(r'\b[a-z]{3,}\b', txt))

        freq = Counter(w for w in words if w not in stop).most_common(12)
        if not freq:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue

        labels, counts = zip(*freq)
        colours_bar = [COLOURS["negative"] if i < 3 else COLOURS["secondary"] if i < 6 else "#95a5a6"
                       for i in range(len(labels))]

        ax.barh(range(len(labels)), counts, color=colours_bar, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_title(band, fontsize=13, fontweight="bold")
        ax.set_xlabel("Occurrences")
        ax.invert_yaxis()

        for i, v in enumerate(counts):
            ax.text(v + 0.15, i, str(v), va="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_fig("10_install_failure_keywords.png")


# ============================================================
# 7. SUMMARY
# ============================================================
def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  INSTALLATION SENTIMENT ANALYSIS — SUMMARY")
    print("="*60)

    total        = len(df)
    with_review  = df["Has_Review"].sum()
    no_review    = total - with_review

    print(f"\n  Total installation responses : {total:,}")
    print(f"  With review text             : {with_review:,}")
    print(f"  Without review text          : {no_review:,}")

    labelled = df[df["Sentiment_Label"] != "No Review"]
    print(f"\n  — Sentiment Breakdown (reviewed only) —")
    for label in ["Positive", "Neutral", "Negative"]:
        count = (labelled["Sentiment_Label"] == label).sum()
        pct   = count / len(labelled) * 100
        print(f"    {label:12s} : {count:5,}  ({pct:5.1f}%)")

    agree = labelled["Agreement"].mean() * 100
    print(f"\n  VADER ↔ TextBlob agreement : {agree:.1f}%")

    avg_compound = labelled["vader_compound"].mean()
    print(f"  Overall avg compound score : {avg_compound:+.3f}")

    # contractor breakdown
    df_adv = df[(df["Advisor"].notna()) & (df["Advisor"].str.strip().str.upper() != "N/A") & df["Has_Review"]].copy()
    if len(df_adv) > 0:
        print(f"\n  — Avg Compound by Contractor (top 10) —")
        contractor_avg = (
            df_adv.groupby("Advisor")["vader_compound"]
            .agg(["mean","count"])
            .sort_values("count", ascending=False)
            .head(10)
            .sort_values("mean", ascending=False)
        )
        for contractor, row in contractor_avg.iterrows():
            print(f"    {contractor:40s} {row['mean']:+.3f}   (n={int(row['count'])})")

    print("\n" + "="*60 + "\n")


def export_results(df: pd.DataFrame):
    export_cols = [
        "Result Date", "C Number", "Department", "Advisor", "Rating",
        "Review", "Percent Score",
        "vader_compound", "vader_pos", "vader_neu", "vader_neg",
        "Sentiment_Label",
        "tb_polarity", "tb_subjectivity", "TB_Label", "Agreement"
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    out_path = "ces_installation_results.xlsx"
    df[export_cols].to_excel(out_path, index=False, engine="openpyxl")
    print(f"  💾 enriched installation results saved → {out_path}\n")


# ============================================================
# 8. MAIN
# ============================================================
def main():
    print("\n🔄 Loading data and filtering to installation reviews …")
    df = load_data(DATA_FILE)

    print("🔄 Running sentiment analysis (VADER + TextBlob) …")
    df = run_sentiment(df)

    print("🔄 Generating installation-specific charts …\n")
    chart_01_overall_distribution(df)
    chart_02_sentiment_by_contractor(df)
    chart_03_avg_score_by_contractor(df)
    chart_04_rating_vs_sentiment(df)
    chart_05_sentiment_trend(df)
    chart_06_vader_vs_textblob(df)
    chart_07_top_reviews(df)
    chart_08_rating_distribution(df)
    chart_09_subjectivity_map(df)
    chart_10_failure_keywords(df)

    print("\n🔄 Summary …")
    print_summary(df)

    print("🔄 Exporting enriched installation data …")
    export_results(df)

    print("✅ All done! Charts saved in  ./{}/  \n".format(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
