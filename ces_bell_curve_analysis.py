"""
============================================================
  CES Bell Curve Analysis
============================================================
  Generates bell curve (normal distribution) visualizations
  for CES ratings and related metrics.
  
  Charts produced (6 total):
    01. Overall Rating Distribution (with fitted normal curve)
    02. Percent Score Distribution (with fitted normal curve)
    03. Rating Distribution by Department (overlaid curves)
    04. Advisor Performance Distribution (avg ratings)
    05. Rating Distribution - Promoters vs Detractors comparison
    06. Multi-Department Bell Curve Comparison Grid
  
  SETUP:
      pip install scipy  (if not already installed)
  
  USAGE:
      python ces_bell_curve_analysis.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_FILE = "data_ces.xlsx"
OUTPUT_FOLDER = "charts_bell_curve"

COLOURS = {
    "primary": "#2980b9",
    "navy": "#1B2845",
    "orange": "#F5A623",
    "teal": "#2A9D8F",
    "coral": "#E76F51",
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral": "#95a5a6",
}

# ============================================================
# HELPERS
# ============================================================
def setup_plot():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

def save_fig(name: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  💾 saved  →  {path}")

def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()
    
    # Parse Rating
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    
    # Parse Percent Score
    if "Percent Score" in df.columns:
        df["Percent Score"] = (
            df["Percent Score"].astype(str)
            .str.replace("%", "", regex=False).str.strip()
        )
        df["Percent Score"] = pd.to_numeric(df["Percent Score"], errors="coerce")
    
    print(f"\n✔ Loaded {len(df):,} responses")
    print(f"  - {df['Rating'].notna().sum():,} with ratings\n")
    
    return df

# ============================================================
# BELL CURVE CHARTS
# ============================================================

# Chart 1: Overall Rating Distribution with Fitted Normal Curve
def chart_01_overall_rating_bell_curve(df: pd.DataFrame):
    setup_plot()
    
    data = df["Rating"].dropna()
    mean = data.mean()
    std = data.std()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("CES Rating Distribution — Bell Curve Analysis", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Histogram
    n, bins, patches = ax.hist(data, bins=7, density=True, alpha=0.7, 
                               color=COLOURS["primary"], edgecolor="white", 
                               linewidth=1.5, label="Actual Distribution")
    
    # Fitted normal curve
    x = np.linspace(data.min(), data.max(), 100)
    fitted_curve = norm.pdf(x, mean, std)
    ax.plot(x, fitted_curve, color=COLOURS["orange"], linewidth=3, 
            label=f"Normal Distribution\n(μ={mean:.2f}, σ={std:.2f})")
    
    # Mean line
    ax.axvline(mean, color=COLOURS["coral"], linestyle="--", linewidth=2, 
               label=f"Mean = {mean:.2f}")
    
    # Standard deviation shading
    ax.axvspan(mean - std, mean + std, alpha=0.15, color=COLOURS["teal"], 
               label=f"±1 SD ({mean-std:.2f} to {mean+std:.2f})")
    ax.axvspan(mean - 2*std, mean + 2*std, alpha=0.08, color=COLOURS["teal"])
    
    # Statistics box
    stats_text = f"""
    Sample Size: {len(data):,}
    Mean: {mean:.2f}
    Median: {data.median():.2f}
    Std Dev: {std:.2f}
    Skewness: {data.skew():.2f}
    
    68% within: [{mean-std:.2f}, {mean+std:.2f}]
    95% within: [{mean-2*std:.2f}, {mean+2*std:.2f}]
    """
    
    ax.text(0.02, 0.97, stats_text.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_xlabel("CES Rating (1-7)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_xlim(0.5, 7.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig("01_overall_rating_bell_curve.png")


# Chart 2: Percent Score Distribution
def chart_02_percent_score_bell_curve(df: pd.DataFrame):
    setup_plot()
    
    data = df["Percent Score"].dropna()
    mean = data.mean()
    std = data.std()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("CES Percent Score Distribution — Bell Curve Analysis", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Histogram
    n, bins, patches = ax.hist(data, bins=20, density=True, alpha=0.7, 
                               color=COLOURS["teal"], edgecolor="white", 
                               linewidth=1.2, label="Actual Distribution")
    
    # Fitted normal curve
    x = np.linspace(data.min(), data.max(), 200)
    fitted_curve = norm.pdf(x, mean, std)
    ax.plot(x, fitted_curve, color=COLOURS["orange"], linewidth=3, 
            label=f"Normal Distribution\n(μ={mean:.1f}%, σ={std:.1f}%)")
    
    # Mean line
    ax.axvline(mean, color=COLOURS["coral"], linestyle="--", linewidth=2, 
               label=f"Mean = {mean:.1f}%")
    
    # Standard deviation shading
    ax.axvspan(mean - std, mean + std, alpha=0.15, color=COLOURS["primary"])
    
    # Statistics box
    stats_text = f"""
    Sample Size: {len(data):,}
    Mean: {mean:.1f}%
    Median: {data.median():.1f}%
    Std Dev: {std:.1f}%
    Skewness: {data.skew():.2f}
    """
    
    ax.text(0.02, 0.97, stats_text.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_xlabel("Percent Score (%)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig("02_percent_score_bell_curve.png")


# Chart 3: Rating Distribution by Department (Overlaid)
def chart_03_department_comparison(df: pd.DataFrame):
    setup_plot()
    
    # Top 5 departments by volume
    top_depts = df["Department"].value_counts().head(5).index.tolist()
    
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Rating Distribution by Department — Bell Curve Comparison", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    colors = [COLOURS["primary"], COLOURS["orange"], COLOURS["teal"], 
              COLOURS["coral"], COLOURS["positive"]]
    
    for idx, dept in enumerate(top_depts):
        data = df[df["Department"] == dept]["Rating"].dropna()
        if len(data) < 30:  # Skip if too few data points
            continue
        
        mean = data.mean()
        std = data.std()
        
        # Fitted curve
        x = np.linspace(1, 7, 100)
        fitted_curve = norm.pdf(x, mean, std)
        
        ax.plot(x, fitted_curve, color=colors[idx], linewidth=2.5, 
                label=f"{dept}\n(μ={mean:.2f}, σ={std:.2f}, n={len(data):,})")
        
        # Mean marker
        ax.axvline(mean, color=colors[idx], linestyle=":", linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel("CES Rating (1-7)", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_xlim(1, 7)
    ax.legend(loc='upper left', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig("03_department_bell_curves.png")


# Chart 4: Advisor Performance Distribution
def chart_04_advisor_performance_distribution(df: pd.DataFrame):
    setup_plot()
    
    # Calculate avg rating per advisor (min 10 responses)
    advisor_avg = (
        df[df["Advisor"].notna()]
        .groupby("Advisor")["Rating"]
        .agg(["mean", "count"])
        .query("count >= 10")["mean"]
    )
    
    if len(advisor_avg) < 10:
        print("  ⚠ Not enough advisors with 10+ responses — skipping chart 04.")
        return
    
    mean = advisor_avg.mean()
    std = advisor_avg.std()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Advisor Performance Distribution — Average CES Ratings", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Histogram
    n, bins, patches = ax.hist(advisor_avg, bins=15, density=True, alpha=0.7, 
                               color=COLOURS["primary"], edgecolor="white", 
                               linewidth=1.2, label="Actual Distribution")
    
    # Fitted normal curve
    x = np.linspace(advisor_avg.min(), advisor_avg.max(), 100)
    fitted_curve = norm.pdf(x, mean, std)
    ax.plot(x, fitted_curve, color=COLOURS["orange"], linewidth=3, 
            label=f"Normal Distribution\n(μ={mean:.2f}, σ={std:.2f})")
    
    # Mean line
    ax.axvline(mean, color=COLOURS["coral"], linestyle="--", linewidth=2, 
               label=f"Mean = {mean:.2f}")
    
    # Performance zones
    ax.axvspan(mean + std, advisor_avg.max(), alpha=0.15, color=COLOURS["positive"], 
               label="High Performers (+1 SD)")
    ax.axvspan(advisor_avg.min(), mean - std, alpha=0.15, color=COLOURS["negative"], 
               label="Low Performers (-1 SD)")
    
    # Statistics box
    stats_text = f"""
    Advisors Analyzed: {len(advisor_avg)}
    Mean Avg Rating: {mean:.2f}
    Std Dev: {std:.2f}
    Range: {advisor_avg.min():.2f} - {advisor_avg.max():.2f}
    
    High Performers (>{mean+std:.2f}): {(advisor_avg > mean + std).sum()}
    Low Performers (<{mean-std:.2f}): {(advisor_avg < mean - std).sum()}
    """
    
    ax.text(0.02, 0.97, stats_text.strip(), transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    ax.set_xlabel("Average CES Rating", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig("04_advisor_performance_bell_curve.png")


# Chart 5: Promoters vs Detractors Comparison
def chart_05_promoters_vs_detractors(df: pd.DataFrame):
    setup_plot()
    
    promoters = df[df["Rating"] >= 6]["Rating"].dropna()
    detractors = df[df["Rating"] <= 3]["Rating"].dropna()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Rating Distribution — Promoters vs Detractors", 
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Promoters curve
    if len(promoters) > 10:
        x_prom = np.linspace(promoters.min(), promoters.max(), 100)
        fitted_prom = norm.pdf(x_prom, promoters.mean(), promoters.std())
        ax.plot(x_prom, fitted_prom, color=COLOURS["positive"], linewidth=3, 
                label=f"Promoters (6-7)\n(μ={promoters.mean():.2f}, n={len(promoters):,})")
        ax.fill_between(x_prom, fitted_prom, alpha=0.2, color=COLOURS["positive"])
    
    # Detractors curve
    if len(detractors) > 10:
        x_det = np.linspace(detractors.min(), detractors.max(), 100)
        fitted_det = norm.pdf(x_det, detractors.mean(), detractors.std())
        ax.plot(x_det, fitted_det, color=COLOURS["negative"], linewidth=3, 
                label=f"Detractors (1-3)\n(μ={detractors.mean():.2f}, n={len(detractors):,})")
        ax.fill_between(x_det, fitted_det, alpha=0.2, color=COLOURS["negative"])
    
    # Passives for reference
    passives = df[(df["Rating"] >= 4) & (df["Rating"] <= 5)]["Rating"].dropna()
    if len(passives) > 10:
        x_pass = np.linspace(passives.min(), passives.max(), 100)
        fitted_pass = norm.pdf(x_pass, passives.mean(), passives.std())
        ax.plot(x_pass, fitted_pass, color=COLOURS["neutral"], linewidth=2, 
                linestyle="--", alpha=0.7,
                label=f"Passives (4-5)\n(μ={passives.mean():.2f}, n={len(passives):,})")
    
    ax.set_xlabel("CES Rating", fontsize=12)
    ax.set_ylabel("Probability Density", fontsize=12)
    ax.set_xlim(0.5, 7.5)
    ax.legend(loc='upper center', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig("05_promoters_vs_detractors_bell_curves.png")


# Chart 6: Multi-Department Grid
def chart_06_department_grid(df: pd.DataFrame):
    setup_plot()
    
    # Top 6 departments
    top_depts = df["Department"].value_counts().head(6).index.tolist()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Department Rating Distributions — Bell Curve Grid", 
                 fontsize=18, fontweight="bold", y=0.98)
    axes = axes.flatten()
    
    for idx, dept in enumerate(top_depts):
        ax = axes[idx]
        data = df[df["Department"] == dept]["Rating"].dropna()
        
        if len(data) < 20:
            ax.text(0.5, 0.5, f"{dept}\n(Insufficient data)", 
                   ha="center", va="center", transform=ax.transAxes)
            continue
        
        mean = data.mean()
        std = data.std()
        
        # Histogram
        ax.hist(data, bins=7, density=True, alpha=0.6, 
               color=COLOURS["primary"], edgecolor="white", linewidth=1)
        
        # Fitted curve
        x = np.linspace(1, 7, 100)
        fitted_curve = norm.pdf(x, mean, std)
        ax.plot(x, fitted_curve, color=COLOURS["orange"], linewidth=2.5)
        
        # Mean line
        ax.axvline(mean, color=COLOURS["coral"], linestyle="--", linewidth=1.5)
        
        # Title with stats
        ax.set_title(f"{dept}\nμ={mean:.2f}, σ={std:.2f}, n={len(data):,}", 
                    fontsize=11, fontweight="bold")
        ax.set_xlabel("Rating", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlim(0.5, 7.5)
        ax.grid(True, alpha=0.2)
    
    # Hide unused subplots
    for idx in range(len(top_depts), 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_fig("06_department_grid_bell_curves.png")


# ============================================================
# SUMMARY STATISTICS
# ============================================================
def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 70)
    print("  BELL CURVE ANALYSIS — SUMMARY")
    print("=" * 70)
    
    data = df["Rating"].dropna()
    mean = data.mean()
    std = data.std()
    
    print(f"\n  Overall Rating Distribution:")
    print(f"    Mean (μ):           {mean:.2f}")
    print(f"    Median:             {data.median():.2f}")
    print(f"    Std Dev (σ):        {std:.2f}")
    print(f"    Skewness:           {data.skew():.2f}")
    print(f"    Kurtosis:           {data.kurtosis():.2f}")
    
    print(f"\n  Normal Distribution Ranges:")
    print(f"    68% of data (±1σ):  [{mean-std:.2f}, {mean+std:.2f}]")
    print(f"    95% of data (±2σ):  [{mean-2*std:.2f}, {mean+2*std:.2f}]")
    print(f"    99.7% of data (±3σ): [{mean-3*std:.2f}, {mean+3*std:.2f}]")
    
    # Actual coverage
    within_1sd = ((data >= mean - std) & (data <= mean + std)).sum() / len(data) * 100
    within_2sd = ((data >= mean - 2*std) & (data <= mean + 2*std)).sum() / len(data) * 100
    
    print(f"\n  Actual Data Coverage:")
    print(f"    Within ±1σ:         {within_1sd:.1f}% (expected: 68%)")
    print(f"    Within ±2σ:         {within_2sd:.1f}% (expected: 95%)")
    
    # Normality test
    if len(data) > 5000:
        sample = data.sample(5000)
    else:
        sample = data
    
    stat, p_value = stats.shapiro(sample)
    print(f"\n  Shapiro-Wilk Normality Test:")
    print(f"    Test Statistic:     {stat:.4f}")
    print(f"    P-value:            {p_value:.4f}")
    print(f"    Result:             {'Normally distributed' if p_value > 0.05 else 'Not normally distributed'} (α=0.05)")
    
    print("\n" + "=" * 70 + "\n")


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n🔄 Loading data …")
    df = load_data(DATA_FILE)
    
    print("🔄 Generating bell curve visualizations …\n")
    chart_01_overall_rating_bell_curve(df)
    chart_02_percent_score_bell_curve(df)
    chart_03_department_comparison(df)
    chart_04_advisor_performance_distribution(df)
    chart_05_promoters_vs_detractors(df)
    chart_06_department_grid(df)
    
    print_summary(df)
    
    print("✅ All bell curve charts generated successfully!")
    print(f"   Output folder: ./{OUTPUT_FOLDER}/\n")


if __name__ == "__main__":
    main()
