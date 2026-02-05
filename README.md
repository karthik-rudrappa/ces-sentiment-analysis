# Customer Effort Score (CES) — Sentiment Analysis & Insights Platform

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production-success.svg)

A comprehensive sentiment analysis platform for Customer Effort Score (CES) feedback, delivering actionable insights across multiple business domains including installations, customer service, loyalty & retention.

---

## 📊 Project Overview

This project performs **multi-layered sentiment analysis** on customer feedback data using a dual-model approach (VADER + TextBlob), combined with keyword-driven analysis and data-driven pair discovery to surface actionable insights for customer experience improvement.

### Key Features

✅ **Dual-Layer Sentiment Analysis** — VADER (social media-tuned) + TextBlob (polarity & subjectivity)  
✅ **Domain-Specific Deep Dives** — Separate analysis pipelines for Installations, Loyalty, and General CES  
✅ **Data-Driven Keyword Discovery** — Automatically extracts top keywords and their contextual pairs from reviews  
✅ **Cross-Department Analysis** — Tracks how keywords surface across different teams  
✅ **14 Production-Ready Visualizations** — Sentiment trends, contractor performance, validation heatmaps, top reviews  
✅ **Churn Risk Detection** — Identifies negative keyword patterns signaling customer attrition  

---

## 🎯 Business Impact

This analysis platform helps organizations:

- **Identify Installation Pain Points** — Contractor performance, appointment issues, damage complaints
- **Predict Churn Risk** — Surface negative patterns in loyalty conversations (pricing, contract issues, competitor mentions)
- **Validate CES Scores** — Cross-check survey ratings against actual sentiment in review text
- **Optimize Resource Allocation** — Identify which teams/advisors are driving positive vs negative sentiment
- **Track Sentiment Trends** — Monitor weekly changes across departments to catch emerging issues early

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Sentiment Analysis** | VADER, TextBlob |
| **NLP** | NLTK (stopwords, tokenization) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Data Source** | Microsoft Dataverse (Excel export) |

---

## 📁 Project Structure

```
CES-Sentiment-Analysis/
│
├── ces_sentiment_analysis_v2.py       # Main keyword-driven analysis
├── ces_installation_analysis.py       # Installation deep-dive
├── ces_loyalty_analysis.py            # Loyalty & retention deep-dive
│
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Protects sensitive data
├── README.md                          # This file
│
└── [Generated Outputs — not tracked in Git]
    ├── charts_v2/                     # Main analysis charts (14 PNGs)
    ├── charts_installation/           # Installation charts (10 PNGs)
    ├── charts_loyalty/                # Loyalty charts (10 PNGs)
    ├── ces_sentiment_results_v2.xlsx  # Enriched main dataset
    ├── ces_installation_results.xlsx  # Enriched installation data
    └── ces_loyalty_results.xlsx       # Enriched loyalty data
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Microsoft Excel (for viewing .xlsx outputs)
- Your CES data exported as `data_ces.xlsx` in the project root

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/karthik-rudrappa/ces-sentiment-analysis.git
   cd ces-sentiment-analysis
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (one-time setup)
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"
   ```

### Usage

**Run the main keyword-driven analysis:**
```bash
python ces_sentiment_analysis_v2.py
```

**Run the installation deep-dive:**
```bash
python ces_installation_analysis.py
```

**Run the loyalty & retention deep-dive:**
```bash
python ces_loyalty_analysis.py
```

Each script generates:
- 10-14 PNG charts in its respective `charts_*/` folder
- An enriched Excel file with sentiment scores attached to every review
- A terminal summary with key metrics

---

## 📈 Analysis Outputs

### Main Analysis (ces_sentiment_analysis_v2.py)

**14 Charts Produced:**

1. **Keyword Sentiment in Customer Service** — Horizontal grouped bar showing positive/neutral/negative breakdown per keyword
2. **Keyword Sentiment in Installation** — Same breakdown for install-related keywords
3. **Top Case Categories by Rating** — Stacked bar chart showing Promoter/Passive/Detractor distribution per department
4. **Sentiment Trend Over Time** — Weekly line chart tracking positive/neutral/negative counts
5. **Avg CES Rating by Department** — Horizontal bar chart with department performance
6. **Negative Keywords by Rating Band** — Top failure themes for Detractors vs Passives
7. **Advisor Performance** — Avg CES rating per advisor (min 2 responses)
8. **Rating Band Distribution** — Pie + bar showing overall NPS-style breakdown
9. **Department Sentiment %** — Stacked horizontal bar showing sentiment proportion per department
10. **Keyword × Department Heatmap** — Avg sentiment score for each keyword in each department
11. **Keyword Mentions Across Departments (Count)** — Heatmap showing where keywords land most
12. **Keyword Sentiment Across Departments** — Same grid but colored by sentiment (green = positive, red = negative)
13. **Keyword Pair Co-occurrence** — Stacked bar showing which keyword pairs appear together (e.g., "Service + Customer")
14. **Keyword Pair Avg Score** — Dot plot ranking pairs by sentiment (most negative at top)

**Key Innovations:**

- **Data-Driven Pair Discovery** — Instead of hardcoding keyword pairs, the script discovers the top 20 keywords from your actual data, then for each keyword finds its top 3 co-occurring partners. This produces contextually meaningful pairs like "Service → Customer, Helpful, Phone" and "Engineer → Time, Problem, Installation".

- **Cross-Department Keyword Tracking** — Chart 11 shows where each keyword appears across departments. For example, "Installation" might appear 340 times in Customer Service, 180 times in Repair, and 1,200 times in Install. This surfaces misrouted cases.

### Installation Analysis (ces_installation_analysis.py)

Filters to installation-related reviews only (Department = "Install" OR review mentions install keywords).

**10 Charts Produced:**

1. Overall sentiment distribution (pie + bar)
2. Sentiment by contractor/install team (stacked horizontal bar)
3. Avg VADER score per contractor
4. Rating vs predicted sentiment validation heatmap
5. Weekly sentiment trend
6. VADER vs TextBlob agreement scatter plot
7. Top 5 positive & negative installation reviews
8. Rating distribution by sentiment (box plot + strip plot)
9. Subjectivity vs polarity map
10. Installation failure keywords by rating band

**Business Value:** Identifies which contractors/teams are driving negative sentiment, surfaces appointment/damage/communication issues, tracks if installation sentiment is improving over time.

### Loyalty & Retention Analysis (ces_loyalty_analysis.py)

Filters to loyalty-related reviews (Department = "Loyalty" OR review mentions cancel/renew/price/contract keywords).

**10 Charts Produced:**

1. Overall sentiment distribution (pie + bar)
2. Sentiment by loyalty advisor (stacked horizontal bar)
3. Avg VADER score per advisor
4. Rating vs predicted sentiment validation heatmap
5. Weekly sentiment trend
6. VADER vs TextBlob agreement scatter plot
7. Top 5 positive & negative loyalty reviews
8. Rating distribution by sentiment (box plot + strip plot)
9. Subjectivity vs polarity map
10. **Churn risk keywords by rating band** — Top negative keywords signaling cancellation intent

**Business Value:** Predicts churn by surfacing keywords like "cancel", "expensive", "contract", "switch". Identifies which advisors are successfully retaining customers vs losing them. Tracks loyalty sentiment trends to catch deterioration early.

---

## 🔍 Methodology

### Sentiment Scoring

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**
- Optimized for short, social media-style text (perfect for CES reviews)
- Returns compound score (−1 to +1)
- Handles punctuation, capitalization, and intensity modifiers ("VERY good" vs "good")
- Thresholds: ≥0.05 = Positive, ≤−0.05 = Negative, else Neutral

**TextBlob**
- Provides polarity (−1 to +1) and subjectivity (0 to 1)
- Used as a cross-validation layer
- Agreement metric calculated between VADER and TextBlob labels

### Keyword Discovery

1. **Corpus Mining** — Extract all words (3+ characters) from reviews, strip stopwords
2. **Frequency Ranking** — Rank by occurrence, filter to words appearing ≥15 times
3. **Top-N Selection** — Keep top 20 keywords (configurable)
4. **Pair Generation** — For each top keyword, find its top 3 co-occurring partners from the same top-20 list
5. **Deduplication** — Alphabetical ordering ensures "Service + Call" and "Call + Service" collapse to one pair

### Data Filtering

**Installation Reviews:**
- Department = "Install" OR
- Review contains: install, engineer, cable, appointment, setup, etc.

**Loyalty Reviews:**
- Department = "Loyalty" OR
- Review contains: cancel, renew, price, contract, switch, competitor names, etc.

---

## 📊 Sample Insights (Anonymized)

> **Finding 1:** "Installation + Delay" co-occurred in 89 reviews with an avg VADER score of −0.42, making it the #1 negative keyword pair. This signals that appointment delays are a critical pain point.

> **Finding 2:** The keyword "service" appeared 1,691 times across all reviews, but when broken down by department, 544 mentions were in Customer Service (avg score +0.18), 129 in Install (avg score −0.05), and 152 in Repair (avg score −0.22). This shows "service" has a different sentiment depending on which team the customer interacted with.

> **Finding 3:** Loyalty reviews mentioning "cancel" or "price" had an avg VADER score of −0.38, while those mentioning "helpful" or "easy" had +0.52. This 0.90-point gap represents the difference between churn risk and successful retention.

> **Finding 4:** Contractor "East Kelly Group Install" had an avg VADER score of +0.35 across 87 reviews, while "South Avonline Install" scored −0.12 across 64 reviews, indicating a significant performance gap between install teams.

---

## 🎨 Customization

### Adjusting Discovery Settings

Edit the configuration block at the top of `ces_sentiment_analysis_v2.py`:

```python
TOP_N_KEYWORDS        = 20   # How many top keywords to discover
MIN_KEYWORD_FREQ      = 15   # Min occurrences for a keyword to qualify
MIN_PAIR_COOCCURRENCE = 3    # Min times a pair must co-occur to be kept
```

### Adding Custom Keywords

For installation or loyalty scripts, extend the keyword lists:

```python
# In ces_installation_analysis.py
INSTALL_KEYWORDS = [
    "install", "installation", "engineer", 
    # Add your custom keywords here
    "your_keyword_1", "your_keyword_2"
]
```

---

## 📦 Data Requirements

Your input data file (`data_ces.xlsx`) should contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Result Date` | Date | When the CES response was recorded |
| `C Number` | String | Customer identifier |
| `Department` | String | Department that handled the interaction |
| `Advisor` | String | Staff member name |
| `Rating` | Integer (1-7) | CES score |
| `Review` | String | Free-text customer feedback |
| `Percent Score` | String | CES as percentage (e.g., "85.71%") |

Additional columns are preserved but not required.

---

## 🔒 Data Privacy

**This repository does NOT contain any customer data.**

The `.gitignore` file blocks:
- All `.xlsx`, `.xls`, `.csv` files
- All generated chart folders (`charts*/`)
- All enriched results files
- Python cache and virtual environments

To use this project:
1. Clone the repository
2. Add your own `data_ces.xlsx` file locally (it will NOT be tracked by Git)
3. Run the scripts — outputs are generated locally and NOT committed

---

## 🤝 Contributing

This is a portfolio project showcasing sentiment analysis and NLP techniques for customer experience data. If you'd like to adapt it for your own use case:

1. Fork the repository
2. Modify the keyword lists and filtering logic for your domain
3. Adjust the chart types/layouts in the visualization functions
4. Share your improvements via pull request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**[Karthik Rudrappa]**  
- LinkedIn: [Karthik Rudrappa](https://linkedin.com/in/karthik-n-rudrappa)  
- GitHub: [Karthik Rudrappa](https://github.com/karthik-rudrappa)  
- Email: krk767@gmail.com

---

## 🙏 Acknowledgments

- **VADER Sentiment** — Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text.
- **TextBlob** — Steven Loria and contributors
- **Python Data Science Stack** — Pandas, NumPy, Matplotlib, Seaborn, NLTK

---

## 📸 Screenshots

*Note: Sample charts are not included in this repository to protect customer data privacy. When running the scripts locally, you'll generate 34 total visualizations across the three analysis pipelines.*

### Sample Chart Types:
- Sentiment distribution pie charts
- Keyword co-occurrence stacked bars
- Contractor/advisor performance dot plots
- Weekly sentiment trend lines
- Rating validation heatmaps
- Top positive/negative review comparisons
- Cross-department keyword heatmaps

---

## 🔮 Future Enhancements

- [ ] Real-time dashboard (Power BI / Tableau integration)
- [ ] Automated email alerts for sentiment threshold breaches
- [ ] Topic modeling (LDA) for unsupervised theme discovery
- [ ] Transformer-based sentiment (BERT/RoBERTa) for comparison
- [ ] API endpoint for live CES scoring
- [ ] Multi-language support

---

**⭐ If you found this project useful, please consider giving it a star on GitHub!**
