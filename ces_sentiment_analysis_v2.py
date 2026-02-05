"""
============================================================
  Customer Effort Score — Keyword-Driven Sentiment Analysis
  (Matched to the Jan Slides approach)
============================================================
  Mirrors the analysis style from the reference presentation:
    - Keyword extraction & sentiment by keyword per department
    - Top case categories / themes by rating
    - Trend lines over time (positive/negative review counts)
    - Stacked bar breakdowns by department & rating band
    - Contractor / Advisor performance views
  
  SETUP (run once in your VS Code terminal):
      pip install -r requirements.txt
      python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
  
  USAGE:
      Place data_ces.xlsx in the same folder as this script, then:
      python ces_sentiment_analysis_v2.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os
import re
from collections import Counter
from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================
DATA_FILE     = "data_ces.xlsx"
OUTPUT_FOLDER = "charts_v2"

# Colour palette — matches the orange/dark tone seen in the reference slides
COLOURS = {
    "orange"   : "#F5A623",
    "dark_org" : "#E8860C",
    "navy"     : "#1B2845",
    "teal"     : "#2A9D8F",
    "coral"    : "#E76F51",
    "muted"    : "#6B7B8D",
    "cream"    : "#F9F5EF",
    "white"    : "#FFFFFF",
    "positive" : "#2A9D8F",   # teal
    "neutral"  : "#F5A623",   # orange
    "negative" : "#E76F51",   # coral
}

# Rating bands — maps the 1-7 CES scale to buckets used in the reference deck
RATING_BANDS = {
    "Promoter (6-7)"  : lambda r: r >= 6,
    "Passive (4-5)"   : lambda r: 4 <= r <= 5,
    "Detractor (1-3)" : lambda r: r <= 3,
}

# Keywords to track per department — based on the keywords highlighted
# in the reference slides. Extend / edit these as you like.
DEPT_KEYWORDS = {
    "Customer Service": ["service", "issue", "call", "time", "waiting", "delay",
                         "wait", "problem", "help", "resolve", "update", "phone"],
    "Install"         : ["communication", "installation", "install", "engineer",
                         "time", "damage", "delay", "waiting", "appointment",
                         "team", "cable", "wifi"],
    "Repair"          : ["repair", "engineer", "time", "delay", "wait", "fix",
                         "problem", "issue", "call", "update", "service"],
    "Sales"           : ["sales", "call", "easy", "information", "setup",
                         "process", "service", "price", "switch"],
    "Loyalty"         : ["service", "call", "wait", "issue", "problem",
                         "resolve", "help", "team", "phone"],
    "Dispatch"        : ["dispatch", "appointment", "time", "delay", "install",
                         "engineer", "schedule", "information", "communication"],
}

# Global keywords tracked across all departments for the top-keywords chart
GLOBAL_KEYWORDS = [
    "service", "issue", "call", "time", "waiting", "delay", "wait",
    "install", "installation", "engineer", "communication", "problem",
    "help", "resolve", "update", "phone", "damage", "appointment",
    "easy", "team", "cable", "wifi", "fix", "information", "process"
]

# ============================================================
# 1b. KEYWORD-PAIR DISCOVERY SETTINGS
# ============================================================
# These control how the script auto-discovers top keywords from your data
# and then auto-generates every possible pair from them.
TOP_N_KEYWORDS        = 20   # how many top keywords to extract from reviews
MIN_KEYWORD_FREQ      = 15   # a keyword must appear at least this many times to qualify
MIN_PAIR_COOCCURRENCE = 3    # a pair must co-occur in at least this many reviews to be kept

# ============================================================
# 2. HELPERS
# ============================================================
def clean_text(text: str) -> str:
    if pd.isna(text) or str(text).strip() == "":
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def get_rating_band(rating: float) -> str:
    if pd.isna(rating):
        return "Unknown"
    for band, fn in RATING_BANDS.items():
        if fn(rating):
            return band
    return "Unknown"

def setup_plot():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family"        : "sans-serif",
        "font.size"          : 10,
        "axes.titlesize"     : 14,
        "axes.titleweight"   : "bold",
        "axes.labelsize"     : 10,
        "figure.facecolor"   : COLOURS["cream"],
        "axes.facecolor"     : COLOURS["white"],
        "axes.spines.top"    : False,
        "axes.spines.right"  : False,
        "axes.edgecolor"     : "#CCC",
    })

def save_fig(name: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLOURS["cream"])
    plt.close()
    print(f"  💾 saved  →  {path}")

# ============================================================
# 3. LOAD & PREPARE
# ============================================================
def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()

    # Percent Score → float
    if "Percent Score" in df.columns:
        df["Percent Score"] = (
            df["Percent Score"].astype(str)
            .str.replace("%", "", regex=False).str.strip()
        )
        df["Percent Score"] = pd.to_numeric(df["Percent Score"], errors="coerce")

    # Rating → numeric
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")

    # Dates
    for col in ["Result Date", "Date Picked"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # Cleaned text + flags
    df["Review_Clean"] = df["Review"].apply(clean_text)
    df["Has_Review"]   = df["Review_Clean"].apply(lambda x: len(x) > 0)
    df["Rating_Band"]  = df["Rating"].apply(get_rating_band)

    print(f"\n✔ Loaded {len(df):,} rows  |  {df['Has_Review'].sum():,} with reviews  |  {(~df['Has_Review']).sum():,} empty\n")
    return df

# ============================================================
# 4. VADER SCORING (kept light — used to colour keyword hits)
# ============================================================
def add_vader(df: pd.DataFrame) -> pd.DataFrame:
    analyser = SentimentIntensityAnalyzer()
    def score(text):
        if not text:
            return np.nan
        return analyser.polarity_scores(text)["compound"]
    df["vader_compound"] = df["Review_Clean"].apply(score)

    def label(c):
        if pd.isna(c):   return "No Review"
        if c >= 0.05:    return "Positive"
        if c <= -0.05:   return "Negative"
        return "Neutral"

    df["Sentiment"] = df["vader_compound"].apply(label)
    print("✔ VADER scoring complete.\n")
    return df

# ============================================================
# 5. KEYWORD EXTRACTION HELPERS
# ============================================================
def keyword_hits(text: str, keywords: list) -> list:
    """Return which keywords appear in text."""
    if not text:
        return []
    words = set(re.findall(r'\b\w+\b', text))
    return [kw for kw in keywords if kw in words]

def keyword_sentiment_table(df: pd.DataFrame, department: str, keywords: list) -> pd.DataFrame:
    """
    For a given department, build a table:
        Keyword | Total Mentions | Positive | Neutral | Negative | Avg Score
    """
    dept_df = df[df["Department"].str.strip().str.lower() == department.lower()].copy()
    dept_df = dept_df[dept_df["Has_Review"]]

    rows = []
    for kw in keywords:
        mask = dept_df["Review_Clean"].str.contains(r'\b' + re.escape(kw) + r'\b', regex=True)
        hits = dept_df[mask]
        if len(hits) == 0:
            continue
        total = len(hits)
        pos   = (hits["Sentiment"] == "Positive").sum()
        neu   = (hits["Sentiment"] == "Neutral").sum()
        neg   = (hits["Sentiment"] == "Negative").sum()
        avg   = hits["vader_compound"].mean()
        rows.append({
            "Keyword"   : kw.title(),
            "Mentions"  : total,
            "Positive"  : pos,
            "Neutral"   : neu,
            "Negative"  : neg,
            "Avg Score" : round(avg, 3),
        })

    return pd.DataFrame(rows).sort_values("Mentions", ascending=False).reset_index(drop=True)


# ── NEW ENGINE A: keyword × department cross-matrix
def cross_dept_keyword_matrix(df: pd.DataFrame, keywords: list) -> pd.DataFrame:
    """
    For every keyword in `keywords`, scan the ENTIRE dataset (all departments).
    Returns one row per (Keyword, Department) combo with Positive / Neutral /
    Negative counts and the average VADER score.
    """
    reviewed = df[df["Has_Review"]].copy()
    rows = []
    for kw in keywords:
        mask = reviewed["Review_Clean"].str.contains(
            r'\b' + re.escape(kw) + r'\b', regex=True
        )
        hits = reviewed[mask]
        if len(hits) == 0:
            continue
        # break down by every department that appears
        for dept, grp in hits.groupby("Department"):
            rows.append({
                "Keyword"   : kw.title(),
                "Department": dept,
                "Total"     : len(grp),
                "Positive"  : (grp["Sentiment"] == "Positive").sum(),
                "Neutral"   : (grp["Sentiment"] == "Neutral").sum(),
                "Negative"  : (grp["Sentiment"] == "Negative").sum(),
                "Avg Score" : round(grp["vader_compound"].mean(), 3),
            })
    return pd.DataFrame(rows)


# ── NEW ENGINE B: keyword-pair co-occurrence scanner
def keyword_pair_cooccurrence(df: pd.DataFrame, pairs: list) -> pd.DataFrame:
    """
    For each (label, [kw1, kw2, ...]) pair, find reviews where ALL keywords
    appear together.  Returns one row per (Label, Department) with sentiment
    counts and avg score — exactly like Engine A but for combos.
    """
    reviewed = df[df["Has_Review"]].copy()
    rows = []
    for label, kws in pairs:
        # build a combined mask: ALL keywords must match
        combined_mask = pd.Series(True, index=reviewed.index)
        for kw in kws:
            combined_mask &= reviewed["Review_Clean"].str.contains(
                r'\b' + re.escape(kw) + r'\b', regex=True
            )
        hits = reviewed[combined_mask]
        if len(hits) == 0:
            continue
        for dept, grp in hits.groupby("Department"):
            rows.append({
                "Pair"      : label,
                "Department": dept,
                "Total"     : len(grp),
                "Positive"  : (grp["Sentiment"] == "Positive").sum(),
                "Neutral"   : (grp["Sentiment"] == "Neutral").sum(),
                "Negative"  : (grp["Sentiment"] == "Negative").sum(),
                "Avg Score" : round(grp["vader_compound"].mean(), 3),
            })
    return pd.DataFrame(rows)


# ============================================================
# 5b. KEYWORD DISCOVERY + PAIR GENERATION  (data-driven)
# ============================================================
from itertools import combinations

# Extended stopword set — generic words that would pollute keyword discovery
EXTRA_STOPS = {
    "", "like", "also", "one", "get", "got", "still", "really", "much",
    "even", "quite", "already", "new", "first", "last", "well", "back",
    "just", "know", "good", "great", "would", "could", "should", "thing",
    "made", "make", "way", "told", "tell", "said", "says", "say",
    "around", "actually", "however", "although", "another", "every",
    "anything", "everything", "nothing", "something", "someone", "everyone",
    "anyone", "really", "quite", "very", "much", "many", "much", "now",
    "then", "than", "been", "being", "were", "was", "will", "going",
    "gone", "took", "take", "took", "given", "give", "took", "come",
    "came", "going", "went", "done", "doing", "need", "needed",
}


def discover_top_keywords(df: pd.DataFrame, top_n: int = TOP_N_KEYWORDS,
                          min_freq: int = MIN_KEYWORD_FREQ) -> list:
    """
    Mine the actual review corpus.  Returns the top_n keywords ranked by
    frequency, after stripping stopwords and short words.
    Prints the discovered list so you can see exactly what the data produced.
    """
    stop = set(stopwords.words("english")) | EXTRA_STOPS

    word_counts = Counter()
    for txt in df.loc[df["Has_Review"], "Review_Clean"]:
        # words of 3+ characters only
        words = re.findall(r'\b[a-z]{3,}\b', txt)
        word_counts.update(w for w in words if w not in stop)

    # filter by minimum frequency, then take top N
    top = [(w, c) for w, c in word_counts.most_common() if c >= min_freq][:top_n]

    keywords = [w for w, _ in top]

    print(f"\n  📝 Discovered top {len(keywords)} keywords from reviews:\n")
    print(f"     {'Keyword':<20s} {'Count':>7s}")
    print(f"     {'─'*20} {'─'*7}")
    for w, c in top:
        print(f"     {w:<20s} {c:>7,}")
    print()

    return keywords


def generate_pairs_from_top(df: pd.DataFrame, top_keywords: list,
                            min_cooccur: int = MIN_PAIR_COOCCURRENCE,
                            pairs_per_anchor: int = 3) -> list:
    """
    HIERARCHICAL PAIR GENERATION (simplified):
    
    For each top keyword (anchor), find which OTHER top keywords co-occur
    most frequently.  Partners are drawn from the SAME top keywords list
    (not the entire corpus), so you get clean, interpretable pairs like:
    
      "Service" → pairs with "Customer", "Helpful", "Call"
      "Engineer" → pairs with "Time", "Problem", "Installation"
    
    Returns the (label, [kw1, kw2]) format that keyword_pair_cooccurrence expects.
    """
    reviewed = df[df["Has_Review"]]
    
    # pre-compute masks for all top keywords
    print(f"  🔍 Pre-computing masks for {len(top_keywords)} keywords …")
    keyword_masks = {}
    for kw in top_keywords:
        keyword_masks[kw] = reviewed["Review_Clean"].str.contains(
            r'\b' + re.escape(kw) + r'\b', regex=True
        )
    
    # for each anchor keyword, find its top N partners (from the same top keywords list)
    print(f"  🔗 For each keyword, finding top {pairs_per_anchor} co-occurring partners …\n")
    
    all_pairs = []
    seen_pairs = set()  # track pairs to avoid duplicates like (A,B) and (B,A)
    
    for anchor in top_keywords:
        anchor_mask = keyword_masks[anchor]
        cooccur_counts = []
        
        # count co-occurrences with every OTHER top keyword
        for partner in top_keywords:
            if partner == anchor:
                continue  # skip self-pairs
            
            count = (anchor_mask & keyword_masks[partner]).sum()
            if count >= min_cooccur:
                cooccur_counts.append((partner, int(count)))
        
        # sort by co-occurrence, take top N
        cooccur_counts.sort(key=lambda x: x[1], reverse=True)
        top_partners = cooccur_counts[:pairs_per_anchor]
        
        # print this anchor's top partners
        if top_partners:
            partners_str = ", ".join(f"{p.title()} ({c}x)" for p, c in top_partners)
            print(f"     {anchor.title():<18s} → {partners_str}")
        
        # build pairs (avoid duplicates with alphabetical ordering)
        for partner, count in top_partners:
            pair_tuple = tuple(sorted([anchor, partner]))
            if pair_tuple not in seen_pairs:
                seen_pairs.add(pair_tuple)
                label = f"{pair_tuple[0].title()} + {pair_tuple[1].title()}"
                all_pairs.append((label, list(pair_tuple), count))
    
    # sort all pairs by co-occurrence count descending
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # format for keyword_pair_cooccurrence
    pairs_out = [(label, kws) for label, kws, _ in all_pairs]
    
    print(f"\n  ✔ Generated {len(pairs_out)} unique pairs from {len(top_keywords)} keywords\n")
    
    # show top 15 pairs overall
    print(f"     {'Top Pairs Overall':<42s} {'Co-occur':>8s}")
    print(f"     {'─'*42} {'─'*8}")
    for (label, _), (_, _, cnt) in zip(pairs_out[:15], all_pairs[:15]):
        print(f"     {label:<42s} {cnt:>8,}")
    if len(pairs_out) > 15:
        print(f"     … and {len(pairs_out)-15} more")
    print()
    
    return pairs_out

def chart_01_keyword_sentiment_cs(df: pd.DataFrame):
    setup_plot()
    tbl = keyword_sentiment_table(df, "Customer Service", DEPT_KEYWORDS["Customer Service"])
    if tbl.empty:
        print("  ⚠ No Customer Service keyword data — skipping chart 01.")
        return
    tbl = tbl.head(8)  # top 8

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Keyword % in Customer Service Reviews", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    y_pos  = np.arange(len(tbl))
    height = 0.25

    ax.barh(y_pos + height, tbl["Positive"], height, label="Positive", color=COLOURS["positive"], edgecolor="white")
    ax.barh(y_pos,          tbl["Neutral"],  height, label="Neutral",  color=COLOURS["neutral"],  edgecolor="white")
    ax.barh(y_pos - height, tbl["Negative"], height, label="Negative", color=COLOURS["negative"], edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tbl["Keyword"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Reviews")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    # value labels
    for i, row in tbl.iterrows():
        idx = list(tbl.index).index(i)
        for val, offset in [(row["Positive"], height), (row["Neutral"], 0), (row["Negative"], -height)]:
            if val > 0:
                ax.text(val + 0.3, idx + offset, str(val), va="center", fontsize=9, fontweight="bold", color=COLOURS["navy"])

    plt.tight_layout()
    save_fig("01_keyword_sentiment_customer_service.png")


# ── 6.2  Top Keywords in Install Reviews
def chart_02_keyword_sentiment_install(df: pd.DataFrame):
    setup_plot()
    tbl = keyword_sentiment_table(df, "Install", DEPT_KEYWORDS["Install"])
    if tbl.empty:
        print("  ⚠ No Install keyword data — skipping chart 02.")
        return
    tbl = tbl.head(8)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Keyword % in Installation Reviews", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    y_pos  = np.arange(len(tbl))
    height = 0.25

    ax.barh(y_pos + height, tbl["Positive"], height, label="Positive", color=COLOURS["positive"], edgecolor="white")
    ax.barh(y_pos,          tbl["Neutral"],  height, label="Neutral",  color=COLOURS["neutral"],  edgecolor="white")
    ax.barh(y_pos - height, tbl["Negative"], height, label="Negative", color=COLOURS["negative"], edgecolor="white")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(tbl["Keyword"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Reviews")
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    for i, row in tbl.iterrows():
        idx = list(tbl.index).index(i)
        for val, offset in [(row["Positive"], height), (row["Neutral"], 0), (row["Negative"], -height)]:
            if val > 0:
                ax.text(val + 0.3, idx + offset, str(val), va="center", fontsize=9, fontweight="bold", color=COLOURS["navy"])

    plt.tight_layout()
    save_fig("02_keyword_sentiment_install.png")


# ── 6.3  Top 5 Case Categories by Rating  (stacked bar — mirrors slide 5 / 6)
def chart_03_top_categories_by_rating(df: pd.DataFrame):
    setup_plot()
    order = ["Promoter (6-7)", "Passive (4-5)", "Detractor (1-3)"]
    pivot = (
        df.groupby(["Department", "Rating_Band"])
        .size().unstack(fill_value=0)
    )
    # keep only the three standard bands
    pivot = pivot.reindex(columns=[c for c in order if c in pivot.columns], fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("Total", ascending=False).head(7).drop(columns="Total")
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=True).index]  # horizontal bar → bottom = highest

    colours_band = {
        "Promoter (6-7)"  : COLOURS["positive"],
        "Passive (4-5)"   : COLOURS["neutral"],
        "Detractor (1-3)" : COLOURS["negative"],
    }

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Top Case Categories by Rating", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    pivot.plot(kind="barh", stacked=True,
               color=[colours_band.get(c, "#999") for c in pivot.columns],
               edgecolor="white", linewidth=1.2, ax=ax, width=0.55)

    ax.set_xlabel("Number of Responses")
    ax.set_ylabel("")
    ax.legend(title="Rating Band", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)

    # percentage labels inside stacked bars
    for i, dept in enumerate(pivot.index):
        total = pivot.loc[dept].sum()
        cum   = 0
        for band in pivot.columns:
            val = pivot.loc[dept, band]
            if val > 0 and (val / total) > 0.05:
                ax.text(cum + val / 2, i, f"{val/total*100:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
            cum += val

    plt.tight_layout()
    save_fig("03_top_categories_by_rating.png")


# ── 6.4  Sentiment Trend Over Time — line chart (mirrors the trend lines in slides 2 & 5)
def chart_04_sentiment_trend(df: pd.DataFrame):
    setup_plot()
    df_t = df[df["Result Date"].notna()].copy()
    df_t["Week"] = df_t["Result Date"].dt.to_period("W").dt.start_time

    trend = (
        df_t[df_t["Sentiment"].isin(["Positive", "Neutral", "Negative"])]
        .groupby(["Week", "Sentiment"])
        .size().unstack(fill_value=0)
    )
    # ensure column order
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in trend.columns:
            trend[col] = 0
    trend = trend[["Positive", "Neutral", "Negative"]]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Sentiment Trend Over Time (Weekly)", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    style_map = {
        "Positive" : {"color": COLOURS["positive"], "ls": "-",  "marker": "o"},
        "Neutral"  : {"color": COLOURS["neutral"],  "ls": "--", "marker": "s"},
        "Negative" : {"color": COLOURS["negative"], "ls": "-.",  "marker": "^"},
    }

    for label in ["Positive", "Neutral", "Negative"]:
        s = style_map[label]
        ax.plot(trend.index, trend[label], color=s["color"], linestyle=s["ls"],
                marker=s["marker"], markersize=5, linewidth=2.2, label=label)
        ax.fill_between(trend.index, trend[label], alpha=0.07, color=s["color"])

    ax.set_xlabel("Week")
    ax.set_ylabel("Count")
    ax.legend(title="Sentiment", framealpha=0.95)
    ax.xaxis.set_tick_params(rotation=35)
    plt.tight_layout()
    save_fig("04_sentiment_trend_over_time.png")


# ── 6.5  Average Case Age / Rating by Department  (grouped bar — mirrors slide 5 chart style)
def chart_05_avg_rating_by_department(df: pd.DataFrame):
    setup_plot()
    avg = df.groupby("Department")["Rating"].agg(["mean", "count"]).sort_values("mean", ascending=True)
    avg = avg[avg["count"] >= 2]  # at least 2 responses

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Average CES Rating by Department", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    bar_colours = [COLOURS["positive"] if v >= 5.5 else COLOURS["neutral"] if v >= 4 else COLOURS["negative"]
                   for v in avg["mean"]]

    bars = ax.barh(avg.index, avg["mean"], color=bar_colours, edgecolor="white", height=0.5, linewidth=1.2)

    # value + count labels
    for bar, (dept, row) in zip(bars, avg.iterrows()):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f"{row['mean']:.2f}  (n={int(row['count'])})",
                va="center", fontsize=10, fontweight="bold", color=COLOURS["navy"])

    ax.set_xlim(0, 7.8)
    ax.set_xlabel("Average Rating (1–7)")
    ax.axvline(df["Rating"].mean(), color=COLOURS["navy"], linewidth=1.2, linestyle="--", alpha=0.5)
    ax.text(df["Rating"].mean() + 0.05, ax.get_ylim()[1] - 0.3,
            f"Overall avg: {df['Rating'].mean():.2f}", fontsize=9, color=COLOURS["navy"], style="italic")

    plt.tight_layout()
    save_fig("05_avg_rating_by_department.png")


# ── 6.6  Top 5 Failure Reasons / Negative Themes by Rating  (keyword cloud per band)
def chart_06_negative_keywords_by_band(df: pd.DataFrame):
    setup_plot()
    neg_df = df[(df["Sentiment"] == "Negative") & df["Has_Review"]].copy()

    # extract top keywords from negative reviews per rating band
    bands = ["Detractor (1-3)", "Passive (4-5)"]
    fig, axes = plt.subplots(1, len(bands), figsize=(15, 5.5), sharey=False)
    fig.patch.set_facecolor(COLOURS["cream"])
    fig.suptitle("Top Negative Keywords by Rating Band", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    stop = set(stopwords.words("english")) | {"", "like", "also", "one", "get", "got", "still", "really", "much", "even", "quite", "already"}

    for ax, band in zip(axes, bands):
        ax.set_facecolor(COLOURS["white"])
        subset = neg_df[neg_df["Rating_Band"] == band]
        words  = []
        for txt in subset["Review_Clean"]:
            words.extend(re.findall(r'\b[a-z]{3,}\b', txt))

        freq = Counter(w for w in words if w not in stop).most_common(12)
        if not freq:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            continue

        labels, counts = zip(*freq)
        colours_bar = [COLOURS["negative"] if i < 3 else COLOURS["coral"] if i < 6 else COLOURS["muted"]
                       for i in range(len(labels))]

        ax.barh(range(len(labels)), counts, color=colours_bar, edgecolor="white", height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.set_title(band, fontsize=13, fontweight="bold", color=COLOURS["navy"])
        ax.set_xlabel("Occurrences")
        ax.invert_yaxis()

        for i, v in enumerate(counts):
            ax.text(v + 0.15, i, str(v), va="center", fontsize=9, fontweight="bold", color=COLOURS["navy"])

    plt.tight_layout()
    save_fig("06_negative_keywords_by_band.png")


# ── 6.7  Advisor / Contractor Performance  (avg rating — mirrors slide 6 contractor chart)
def chart_07_advisor_performance(df: pd.DataFrame):
    setup_plot()
    # filter out N/A and blanks
    adv = df[(df["Advisor"].notna()) & (df["Advisor"].str.strip().str.upper() != "N/A")].copy()
    adv["Advisor"] = adv["Advisor"].str.strip()

    perf = adv.groupby("Advisor")["Rating"].agg(["mean", "count"]).sort_values("count", ascending=False)
    perf = perf[perf["count"] >= 2].head(12).sort_values("mean", ascending=True)  # top 12 by volume

    fig, ax = plt.subplots(figsize=(13, max(5, len(perf)*0.42)))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Advisor Performance — Avg CES Rating  (min 2 responses)", fontsize=15, fontweight="bold", color=COLOURS["navy"])

    bar_colours = [COLOURS["positive"] if v >= 5.5 else COLOURS["neutral"] if v >= 4 else COLOURS["negative"]
                   for v in perf["mean"]]

    bars = ax.barh(perf.index, perf["mean"], color=bar_colours, edgecolor="white", height=0.55)

    for bar, (adv_name, row) in zip(bars, perf.iterrows()):
        ax.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                f"{row['mean']:.2f}  (n={int(row['count'])})",
                va="center", fontsize=9, fontweight="bold", color=COLOURS["navy"])

    ax.set_xlim(0, 8)
    ax.set_xlabel("Average Rating (1–7)")
    ax.axvline(df["Rating"].mean(), color=COLOURS["navy"], linewidth=1, linestyle="--", alpha=0.4)

    plt.tight_layout()
    save_fig("07_advisor_performance.png")


# ── 6.8  Rating Band Distribution — overall pie  (quick exec summary visual)
def chart_08_rating_band_pie(df: pd.DataFrame):
    setup_plot()
    counts = df["Rating_Band"].value_counts()
    order  = ["Promoter (6-7)", "Passive (4-5)", "Detractor (1-3)"]
    counts = counts.reindex([o for o in order if o in counts.index], fill_value=0)
    colours_pie = [COLOURS["positive"], COLOURS["neutral"], COLOURS["negative"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLOURS["cream"])
    fig.suptitle("CES Rating Band Distribution", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    # ── pie
    wedges, texts, auts = ax1.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colours_pie, startangle=130, explode=[0.03]*len(counts), shadow=True
    )
    for at in auts:
        at.set_fontweight("bold")
        at.set_fontsize(11)
    ax1.set_title("Proportion", fontsize=13, color=COLOURS["navy"])

    # ── bar with counts
    bars = ax2.bar(counts.index, counts.values, color=colours_pie, edgecolor="white", width=0.5, linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", fontweight="bold", fontsize=13, color=COLOURS["navy"])
    ax2.set_ylabel("Count")
    ax2.set_title("Counts", fontsize=13, color=COLOURS["navy"])
    ax2.set_ylim(0, counts.max() * 1.18)
    ax2.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    save_fig("08_rating_band_distribution.png")


# ── 6.9  Department Sentiment Stacked Bar  (positive/neutral/negative %)
def chart_09_department_sentiment_stacked(df: pd.DataFrame):
    setup_plot()
    order_sent = ["Positive", "Neutral", "Negative"]
    pivot = (
        df[df["Sentiment"].isin(order_sent)]
        .groupby(["Department", "Sentiment"])
        .size().unstack(fill_value=0)[order_sent]
    )
    # percentage view
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.sort_values("Positive", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(5, len(pivot_pct)*0.45)))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Sentiment % by Department", fontsize=16, fontweight="bold", color=COLOURS["navy"])

    pivot_pct.plot(kind="barh", stacked=True,
                   color=[COLOURS["positive"], COLOURS["neutral"], COLOURS["negative"]],
                   edgecolor="white", linewidth=1.2, ax=ax, width=0.58)

    ax.set_xlabel("Percentage (%)")
    ax.set_ylabel("")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)
    ax.set_xlim(0, 105)

    # labels inside bars
    for i, dept in enumerate(pivot_pct.index):
        cum = 0
        for sent in order_sent:
            val = pivot_pct.loc[dept, sent]
            if val > 5:
                ax.text(cum + val/2, i, f"{val:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
            cum += val

    plt.tight_layout()
    save_fig("09_department_sentiment_pct.png")


# ── 6.10 Keyword Avg Score Heatmap — all departments × top global keywords
def chart_10_keyword_heatmap(df: pd.DataFrame):
    setup_plot()
    depts = df["Department"].dropna().unique()
    top_kw = GLOBAL_KEYWORDS[:15]

    matrix = []
    for dept in depts:
        row = {}
        dept_df = df[(df["Department"].str.strip() == dept) & df["Has_Review"]]
        for kw in top_kw:
            mask = dept_df["Review_Clean"].str.contains(r'\b' + re.escape(kw) + r'\b', regex=True)
            hits = dept_df[mask]
            row[kw.title()] = hits["vader_compound"].mean() if len(hits) > 0 else np.nan
        matrix.append(row)

    heatmap_df = pd.DataFrame(matrix, index=depts)

    fig, ax = plt.subplots(figsize=(14, max(5, len(depts)*0.55)))
    fig.patch.set_facecolor(COLOURS["cream"])
    fig.suptitle("Avg Sentiment Score — Keywords × Departments\n(Green = Positive  |  Red = Negative  |  Grey = No Data)",
                 fontsize=15, fontweight="bold", color=COLOURS["navy"])

    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                ax=ax, linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Avg Compound Score", "shrink": 0.8},
                annot_kws={"size": 9}, vmin=-1, vmax=1,
                mask=heatmap_df.isna())

    ax.set_xlabel("Keyword", fontsize=11)
    ax.set_ylabel("Department", fontsize=11)
    plt.tight_layout()
    save_fig("10_keyword_sentiment_heatmap.png")


# ── 6.11  Keyword × Department — COUNT heatmap
#          "Installation appears in Customer Service THIS many times"
def chart_11_keyword_dept_count_heatmap(df: pd.DataFrame):
    setup_plot()
    cross = cross_dept_keyword_matrix(df, GLOBAL_KEYWORDS)
    if cross.empty:
        print("  ⚠ No cross-dept keyword data — skipping chart 11.")
        return

    # pivot: rows = keywords, columns = departments, values = Total mentions
    pivot = cross.pivot_table(index="Keyword", columns="Department",
                              values="Total", aggfunc="sum", fill_value=0)
    # keep only keywords that have at least 10 total mentions, then sort by total descending
    pivot = pivot[pivot.sum(axis=1) >= 10]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    pivot = pivot.head(14)  # top 14 keywords by total volume

    fig, ax = plt.subplots(figsize=(15, max(7, len(pivot) * 0.5)))
    fig.patch.set_facecolor(COLOURS["cream"])
    fig.suptitle("Keyword Mentions Across Departments\n(How many times each keyword lands in each department)",
                 fontsize=15, fontweight="bold", color=COLOURS["navy"])

    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax,
                linewidths=0.6, linecolor="white",
                cbar_kws={"label": "Mention Count", "shrink": 0.85},
                annot_kws={"size": 10, "weight": "bold"})

    ax.set_xlabel("Department", fontsize=11)
    ax.set_ylabel("Keyword", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_fig("11_keyword_dept_count_heatmap.png")


# ── 6.12  Keyword × Department — SENTIMENT heatmap (avg VADER score)
#          Same grid, but colour = how positive/negative the reviews are
def chart_12_keyword_dept_sentiment_heatmap(df: pd.DataFrame):
    setup_plot()
    cross = cross_dept_keyword_matrix(df, GLOBAL_KEYWORDS)
    if cross.empty:
        print("  ⚠ No cross-dept keyword data — skipping chart 12.")
        return

    pivot = cross.pivot_table(index="Keyword", columns="Department",
                              values="Avg Score", aggfunc="mean")

    # same keyword filter as chart 11 — use count pivot to decide which to keep
    count_pivot = cross.pivot_table(index="Keyword", columns="Department",
                                    values="Total", aggfunc="sum", fill_value=0)
    keep = count_pivot[count_pivot.sum(axis=1) >= 10].index
    pivot = pivot.loc[pivot.index.intersection(keep)]
    pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]  # most negative at top
    pivot = pivot.head(14)

    fig, ax = plt.subplots(figsize=(15, max(7, len(pivot) * 0.5)))
    fig.patch.set_facecolor(COLOURS["cream"])
    fig.suptitle("Keyword Sentiment Score Across Departments\n(Green = Positive  |  Red = Negative  |  White = No Data)",
                 fontsize=15, fontweight="bold", color=COLOURS["navy"])

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", center=0,
                ax=ax, linewidths=0.6, linecolor="white",
                cbar_kws={"label": "Avg VADER Score", "shrink": 0.85},
                annot_kws={"size": 10, "weight": "bold"},
                vmin=-1, vmax=1, mask=pivot.isna())

    ax.set_xlabel("Department", fontsize=11)
    ax.set_ylabel("Keyword", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_fig("12_keyword_dept_sentiment_heatmap.png")


# ── 6.13  Keyword Pairs — co-occurrence by department (grouped stacked bars)
#          "Installation + Delay together — which depts do they land in, pos/neg?"
def chart_13_keyword_pairs_by_dept(pairs_df: pd.DataFrame):
    setup_plot()
    if pairs_df.empty:
        print("  ⚠ No keyword-pair co-occurrence data — skipping chart 13.")
        return

    # aggregate: for each Pair, total across all depts
    pair_totals = pairs_df.groupby("Pair")["Total"].sum().sort_values(ascending=False)
    # keep only pairs that actually fired (total >= 3)
    top_pairs = pair_totals[pair_totals >= 3].head(12).index.tolist()
    pairs_df  = pairs_df[pairs_df["Pair"].isin(top_pairs)]

    if pairs_df.empty:
        print("  ⚠ No keyword pairs with ≥3 co-occurrences — skipping chart 13.")
        return

    # For each pair, build: Positive / Neutral / Negative totals (summed across depts)
    summary = (
        pairs_df.groupby("Pair")[["Positive", "Neutral", "Negative"]]
        .sum()
        .loc[top_pairs]   # preserve the sorted order
    )
    summary = summary.loc[summary.sum(axis=1).sort_values(ascending=True).index]

    fig, ax = plt.subplots(figsize=(14, max(6, len(summary) * 0.52)))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Keyword Pair Co-occurrence — Sentiment Breakdown\n(Reviews where BOTH keywords appear together)",
                 fontsize=15, fontweight="bold", color=COLOURS["navy"])

    summary.plot(kind="barh", stacked=True,
                 color=[COLOURS["positive"], COLOURS["neutral"], COLOURS["negative"]],
                 edgecolor="white", linewidth=1.2, ax=ax, width=0.6)

    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left", framealpha=0.95)

    # inline % labels
    for i, pair in enumerate(summary.index):
        total = summary.loc[pair].sum()
        cum   = 0
        for sent in ["Positive", "Neutral", "Negative"]:
            val = summary.loc[pair, sent]
            if val > 0 and (val / total) > 0.07:
                ax.text(cum + val / 2, i, f"{val/total*100:.0f}%",
                        ha="center", va="center", fontsize=9,
                        fontweight="bold", color="white")
            cum += val
        # total count at the end of the bar
        ax.text(total + 0.3, i, str(int(total)),
                va="center", fontsize=9, fontweight="bold", color=COLOURS["navy"])

    plt.tight_layout()
    save_fig("13_keyword_pairs_sentiment.png")


# ── 6.14  Keyword Pairs — Avg Score dot plot (ranked most negative first)
#          Quick "which combo is the biggest pain point?" view
def chart_14_keyword_pairs_avg_score(pairs_df: pd.DataFrame):
    setup_plot()
    if pairs_df.empty:
        print("  ⚠ No keyword-pair co-occurrence data — skipping chart 14.")
        return

    # weighted avg score per pair (across all depts)
    agg = (
        pairs_df.groupby("Pair")
        .apply(lambda g: np.average(g["Avg Score"], weights=g["Total"]))
        .rename("Avg Score")
        .to_frame()
    )
    agg["Total"] = pairs_df.groupby("Pair")["Total"].sum()
    agg = agg[agg["Total"] >= 3].sort_values("Avg Score", ascending=True).head(15)

    if agg.empty:
        print("  ⚠ No keyword pairs with ≥3 hits — skipping chart 14.")
        return

    fig, ax = plt.subplots(figsize=(13, max(5, len(agg) * 0.48)))
    fig.patch.set_facecolor(COLOURS["cream"])
    ax.set_facecolor(COLOURS["white"])
    fig.suptitle("Keyword Pair Avg Sentiment Score\n(Most negative pain-points at the top)",
                 fontsize=15, fontweight="bold", color=COLOURS["navy"])

    dot_colours = [
        COLOURS["negative"] if v <= -0.05 else COLOURS["neutral"] if v < 0.05 else COLOURS["positive"]
        for v in agg["Avg Score"]
    ]

    ax.barh(agg.index, agg["Avg Score"], color=dot_colours, edgecolor="white", height=0.55)
    ax.axvline(0, color=COLOURS["navy"], linewidth=1, linestyle="--", alpha=0.45)

    # value + n labels
    for i, (pair, row) in enumerate(agg.iterrows()):
        offset = 0.015 if row["Avg Score"] >= 0 else -0.015
        ha     = "left"  if row["Avg Score"] >= 0 else "right"
        ax.text(row["Avg Score"] + offset, i,
                f"{row['Avg Score']:+.3f}  (n={int(row['Total'])})",
                va="center", ha=ha, fontsize=9, fontweight="bold", color=COLOURS["navy"])

    ax.set_xlabel("Avg VADER Compound Score  (−1 → +1)")
    ax.set_xlim(min(agg["Avg Score"].min() - 0.18, -0.15),
                max(agg["Avg Score"].max() + 0.18,  0.15))
    plt.tight_layout()
    save_fig("14_keyword_pairs_avg_score.png")


# ============================================================
# 7. SUMMARY + EXPORT
# ============================================================
def print_summary(df: pd.DataFrame, pairs_df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("   KEYWORD-DRIVEN SENTIMENT ANALYSIS — SUMMARY")
    print("=" * 65)

    total       = len(df)
    reviewed    = df["Has_Review"].sum()
    print(f"\n  Total responses          : {total:,}")
    print(f"  With review text         : {reviewed:,}")
    print(f"  Without review text      : {total - reviewed:,}")

    # rating bands
    print(f"\n  ── Rating Band Breakdown ──")
    for band in ["Promoter (6-7)", "Passive (4-5)", "Detractor (1-3)"]:
        n   = (df["Rating_Band"] == band).sum()
        pct = n / total * 100
        print(f"    {band:22s} : {n:5,}  ({pct:5.1f}%)")

    # sentiment
    labelled = df[df["Sentiment"] != "No Review"]
    print(f"\n  ── Sentiment Breakdown (reviewed only) ──")
    for s in ["Positive", "Neutral", "Negative"]:
        n   = (labelled["Sentiment"] == s).sum()
        pct = n / len(labelled) * 100
        print(f"    {s:12s} : {n:5,}  ({pct:5.1f}%)")

    # per-department keyword summary
    print(f"\n  ── Top Keyword per Department (most mentioned, negative) ──")
    stop = set(stopwords.words("english")) | {"", "like", "also", "one", "get", "got", "still", "really", "much", "even", "quite", "already"}
    for dept in df["Department"].dropna().unique():
        neg = df[(df["Department"] == dept) & (df["Sentiment"] == "Negative") & df["Has_Review"]]
        if len(neg) == 0:
            continue
        words = []
        for txt in neg["Review_Clean"]:
            words.extend(re.findall(r'\b[a-z]{4,}\b', txt))
        freq = Counter(w for w in words if w not in stop)
        if freq:
            top_word, top_count = freq.most_common(1)[0]
            print(f"    {dept:40s} → \"{top_word}\"  ({top_count}x in negative reviews)")

    # ── keyword-pair co-occurrence summary
    print(f"\n  ── Top Keyword Pairs (co-occurring, by volume) ──")
    if not pairs_df.empty:
        pair_agg = (
            pairs_df.groupby("Pair")
            .agg(Total=("Total", "sum"),
                 Positive=("Positive", "sum"),
                 Negative=("Negative", "sum"))
            .sort_values("Total", ascending=False)
            .head(10)
        )
        for pair, row in pair_agg.iterrows():
            neg_pct = row["Negative"] / row["Total"] * 100 if row["Total"] > 0 else 0
            print(f"    {pair:42s} n={int(row['Total']):5,}   neg={neg_pct:5.1f}%")
    else:
        print("    (no pairs fired with ≥1 co-occurrence)")

    print("\n" + "=" * 65 + "\n")


def export_results(df: pd.DataFrame):
    export_cols = [
        "Result Date", "C Number", "Department", "Advisor", "Rating",
        "Review", "Percent Score", "Rating_Band",
        "vader_compound", "Sentiment"
    ]
    export_cols = [c for c in export_cols if c in df.columns]
    out = "ces_sentiment_results_v2.xlsx"
    df[export_cols].to_excel(out, index=False, engine="openpyxl")
    print(f"  💾 enriched results saved → {out}\n")


# ============================================================
# 8. MAIN
# ============================================================
def main():
    print("\n🔄 Loading data …")
    df = load_data(DATA_FILE)

    print("🔄 Running VADER scoring …")
    df = add_vader(df)

    # ── keyword discovery + pair generation (data-driven, runs once)
    print("🔄 Discovering top keywords from your reviews …")
    top_keywords = discover_top_keywords(df)

    print("🔄 Generating keyword pairs from discovered top keywords …")
    discovered_pairs = generate_pairs_from_top(df, top_keywords)

    print("🔄 Scoring keyword-pair co-occurrences …")
    pairs_df = keyword_pair_cooccurrence(df, discovered_pairs)

    # ── charts 1–12  (unchanged)
    print("🔄 Generating charts …\n")
    chart_01_keyword_sentiment_cs(df)
    chart_02_keyword_sentiment_install(df)
    chart_03_top_categories_by_rating(df)
    chart_04_sentiment_trend(df)
    chart_05_avg_rating_by_department(df)
    chart_06_negative_keywords_by_band(df)
    chart_07_advisor_performance(df)
    chart_08_rating_band_pie(df)
    chart_09_department_sentiment_stacked(df)
    chart_10_keyword_heatmap(df)
    chart_11_keyword_dept_count_heatmap(df)
    chart_12_keyword_dept_sentiment_heatmap(df)

    # ── charts 13–14  (pass in the pre-computed pairs_df)
    chart_13_keyword_pairs_by_dept(pairs_df)
    chart_14_keyword_pairs_avg_score(pairs_df)

    print("\n🔄 Summary …")
    print_summary(df, pairs_df)

    print("🔄 Exporting enriched data …")
    export_results(df)

    print("✅ All done! Charts saved in  ./{}/  \n".format(OUTPUT_FOLDER))


if __name__ == "__main__":
    main()
