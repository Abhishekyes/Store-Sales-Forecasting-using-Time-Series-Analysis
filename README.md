<div align="center">

# 🏪 Store Sales Forecasting Using Time Series Analysis

### Predicting Sales for Thousands of Product Families at Favorita Stores in Ecuador

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-9ACD32?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-EC4E20?style=for-the-badge)](https://xgboost.readthedocs.io)

---

</div>

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Description](#-dataset-description)
- [Ecuador-Specific Domain Knowledge](#-ecuador-specific-domain-knowledge)
- [Project Pipeline](#-project-pipeline)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Models Implemented](#-models-implemented)
- [Evaluation Metric](#-evaluation-metric)
- [Key Findings](#-key-findings)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 🎯 Project Overview

This project tackles the **Kaggle Store Sales — Time Series Forecasting** competition. The objective is to build machine learning models that accurately predict **unit sales** for thousands of product families sold at **Corporación Favorita** — a large grocery retailer based in Ecuador.

The dataset spans **January 2013 to August 2017** (~3 million training records) across **54 stores** and **33 product families**, making it a rich, multi-dimensional time series forecasting problem. Our approach combines robust exploratory data analysis, domain-specific feature engineering, and ensemble gradient boosting models to achieve competitive predictions.

---

## 🧩 Problem Statement

> **Goal:** Predict the `sales` for each `(store_nbr, family)` pair for 15 consecutive days (August 16–31, 2017) immediately following the training period.

Sales prediction is critical for retail operations — it drives inventory management, staffing decisions, promotional strategies, and supply chain optimization. The challenge is amplified by Ecuador-specific economic factors such as oil price dependency, earthquake aftermath effects, and unique holiday transfer mechanisms.

---

## 📊 Dataset Description

The competition provides **7 data files** totaling over **120 MB**:

| File | Records | Description |
|:-----|--------:|:------------|
| `train.csv` | 3,000,888 | Historical sales data (2013-01-01 to 2017-08-15) with `store_nbr`, `family`, `sales`, and `onpromotion` |
| `test.csv` | 28,512 | 15-day forecast window (2017-08-16 to 2017-08-31) — same features, predict `sales` |
| `stores.csv` | 54 | Store metadata: `city`, `state`, `type` (A–E), `cluster` (1–17) |
| `oil.csv` | 1,218 | Daily oil prices (`dcoilwtico`) — Ecuador's economy is oil-dependent |
| `holidays_events.csv` | 350 | Holidays & events with transfer/bridge logic and locale info |
| `transactions.csv` | 83,488 | Daily transaction counts per store (2013-01-01 to 2017-08-15) |
| `sample_submission.csv` | 28,512 | Correct submission format |

### Key Data Characteristics

- **54 stores** across **22 cities** and **16 states** in Ecuador
- **33 product families** (e.g., GROCERY, BEVERAGES, PRODUCE, DAIRY, etc.)
- **31.3% zero-sales records** — indicating closed stores on holidays or zero demand
- **No negative sales** in the dataset
- **No missing values** in the core training data
- **Oil price data** has 43 missing values (handled via interpolation)

---

## 🌎 Ecuador-Specific Domain Knowledge

Understanding Ecuador's economic and cultural context was crucial for accurate feature engineering:

| Factor | Description | Impact on Sales |
|:-------|:------------|:----------------|
| 🛢️ **Oil Dependency** | Ecuador's economy is heavily reliant on oil exports | Oil price fluctuations directly affect consumer spending |
| 💰 **Payday Effect** | Public sector wages are paid on the **15th** and **last day** of each month | Sales spike on and around paydays |
| 🌍 **2016 Earthquake** | Magnitude 7.8 earthquake on **April 16, 2016** | People donated water & essentials, disrupting grocery sales for weeks |
| 📅 **Transferred Holidays** | Government can officially move a holiday to a different date | Creates unexpected demand shifts |
| 🌉 **Bridge Days** | Extra days added around holidays to create long weekends | Extended holiday periods boost or suppress sales |
| 📆 **Work Days** | Compensatory work days for bridge days | Sales patterns differ from regular weekdays |
| 🎄 **Christmas Effect** | End-of-year celebrations with December bonuses | Massive annual sales spikes in December |

---

## 🔄 Project Pipeline

Our structured approach follows the full data science lifecycle:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  1. Data Loading │────▶│  2. Data         │────▶│  3. Exploratory     │
│  & Inspection    │     │  Preprocessing   │     │  Data Analysis      │
└─────────────────┘     └─────────────────┘     └─────────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
│  6. Submission   │◀────│  5. Model        │◀────│  4. Feature         │
│  Generation      │     │  Training &      │     │  Engineering        │
│                  │     │  Evaluation      │     │  (30+ features)     │
└─────────────────┘     └─────────────────┘     └─────────────────────┘
```

### Detailed Steps:

1. **Data Loading & Inspection** — Load all 7 CSV files, check data types, missing values, and basic statistics
2. **Data Preprocessing** — Handle missing oil prices (interpolation + backfill), encode categorical variables, merge auxiliary datasets
3. **Exploratory Data Analysis (EDA)** — Comprehensive visual analysis of trends, seasonality, earthquake impact, oil correlation, store/product behavior
4. **Feature Engineering** — Construct 30+ predictive features from temporal patterns, lag values, rolling statistics, and external data
5. **Model Training & Evaluation** — Train baseline (Random Forest) and advanced models (LightGBM, XGBoost) with multiple validation intervals
6. **Submission Generation** — Recursive day-by-day forecasting for the 15-day test window

---

## 📈 Exploratory Data Analysis

Our EDA revealed critical patterns in the data:

### Overall Sales Trend
- **Clear upward trend** in total daily sales from 2013 to 2017
- **Strong December spikes** every year (Christmas season)
- **Earthquake disruption** visible in mid-April 2016, with recovery over subsequent weeks

### Earthquake Impact Analysis
- The **April 16, 2016 earthquake** caused an immediate spike in sales (relief donations of essentials)
- Followed by a prolonged period of altered purchasing behavior
- Sales patterns took **several weeks** to normalize

### Weekly & Monthly Patterns
- **Saturday** and **Sunday** show distinct patterns — Sunday is typically higher
- **End-of-month** and **mid-month** peaks correlate with paydays
- **Day-of-week seasonality** is strong and consistent across years

### Oil Price Correlation
- **Negative correlation** between oil prices and total sales
- Oil price drops often precede economic uncertainty, affecting consumer spending

### Product Family Analysis
- **GROCERY I** dominates sales volume (largest product family)
- **BEVERAGES**, **PRODUCE**, and **CLEANING** are also top contributors
- Some families (e.g., **BOOKS**, **BABY CARE**) show very low and sparse sales

### Store Analysis
- **Type D** stores (18 stores) are the most common
- **Quito** (capital city) has 18 stores — the highest concentration
- Significant variation in average sales across store types and clusters

### Promotion Effect
- Promotions have a **strong positive impact** on sales
- The effect varies by product family — groceries and beverages respond most strongly

---

## ⚙️ Feature Engineering

We engineered **30+ features** from multiple data sources to capture temporal, economic, and domain-specific patterns:

### Temporal Features
| Feature | Description |
|:--------|:------------|
| `year`, `month`, `day` | Basic calendar decomposition |
| `day_of_week`, `week_of_year` | Weekly seasonality signals |
| `day_of_month`, `day_of_year` | Monthly/annual position |
| `quarter` | Quarterly business cycles |
| `is_weekend` | Saturday/Sunday indicator |
| `is_month_start`, `is_month_end` | Month boundary flags |

### Economic Features
| Feature | Description |
|:--------|:------------|
| `oil_price` | Daily oil price (interpolated) |
| `oil_ma_7`, `oil_ma_30` | 7-day and 30-day oil price moving averages |
| `oil_pct_change` | Oil price percentage change |

### Payday & Domain Features
| Feature | Description |
|:--------|:------------|
| `is_payday` | Flag for 15th and last day of month |
| `days_to_payday` | Distance to nearest payday |
| `is_earthquake_period` | Earthquake disruption window flag |

### Holiday Features
| Feature | Description |
|:--------|:------------|
| `is_holiday` | National holiday indicator |
| `holiday_type` | Holiday category (Holiday, Event, Transfer, Bridge, Work Day, Additional) |
| `is_transferred` | Whether the holiday was transferred |

### Lag & Rolling Features
| Feature | Description |
|:--------|:------------|
| `sales_lag_7/14/30` | Lagged sales at 7, 14, and 30-day intervals |
| `sales_rolling_mean_7/14/30` | Rolling mean sales over 7, 14, and 30-day windows |
| `sales_rolling_std_7` | Rolling standard deviation (volatility measure) |

### Store & Product Features
| Feature | Description |
|:--------|:------------|
| `store_type`, `store_cluster` | Store metadata |
| `city_encoded`, `state_encoded` | Location encoding |
| `family_encoded` | Product family encoding |

---

## 🤖 Models Implemented

### 1. Baseline — Random Forest Regressor
- **Purpose:** Establish a performance benchmark
- **Configuration:** `n_estimators=100`, `max_depth=15`, `n_jobs=-1`
- **Role:** Quick baseline model to gauge feature quality

### 2. LightGBM (Light Gradient Boosting Machine)
- **Purpose:** Primary production model — fast and accurate for tabular data
- **Key Advantages:**
  - Handles categorical features natively
  - Leaf-wise tree growth for better accuracy
  - Built-in support for missing values
  - Fast training on large datasets (~3M rows)
- **Validation Strategy:** Tested at **7-day**, **15-day**, and **30-day** forecast intervals

### 3. XGBoost (Extreme Gradient Boosting)
- **Purpose:** Secondary model for comparison and ensemble potential
- **Configuration:** Tuned hyperparameters using validation splits
- **RMSLE loss:** Custom objective for competition metric alignment

### Validation Strategy
- **Time-based split** — train on older data, validate on most recent period
- **Multiple horizons** — 7, 15, and 30-day validation windows
- **Recursive forecasting** — Day-by-day prediction for the 15-day test window, feeding predictions back as lag features

---

## 📏 Evaluation Metric

The competition uses **RMSLE** (Root Mean Squared Logarithmic Error):

$$RMSLE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(1 + \hat{y}_i) - \log(1 + y_i))^2}$$

**Why RMSLE?**
- Penalizes under-prediction more heavily than over-prediction
- Scale-invariant — treats errors proportionally
- Robust to outliers in high-sales products
- Appropriate for sales data where relative accuracy matters more than absolute error

---

## 🔑 Key Findings

1. **Temporal patterns are dominant** — Day-of-week, payday cycles, and December seasonality are the strongest predictors
2. **Oil price is a meaningful signal** — Particularly when used as rolling averages and percentage changes
3. **Lag features are critical** — 7-day and 14-day lag sales capture short-term momentum effectively
4. **Earthquake created a regime change** — Models need explicit flagging of this anomalous period
5. **Product families behave differently** — GROCERY I accounts for the majority of sales; niche families need separate handling
6. **Store clusters reveal geographic patterns** — Location-based features add meaningful predictive power
7. **Promotions strongly boost sales** — Especially for grocery and beverage categories
8. **LightGBM outperforms** — Faster training and comparable or better accuracy vs. XGBoost on this dataset

---

## 📁 Project Structure

```
The store sales- forecasting using time series/
│
├── README.md                                  # Project documentation (this file)
│
├── store_sales_complete_analysis.ipynb        # Complete analysis notebook (detailed version)
│
├── Notebook/
│   └── store-sales-time-series-f.ipynb        # Kaggle submission notebook
│
├── Store sales kaggle/                        # Dataset directory
│   ├── train.csv                              # Training data (~122 MB, 3M rows)
│   ├── test.csv                               # Test data (28,512 rows)
│   ├── stores.csv                             # Store metadata (54 stores)
│   ├── oil.csv                                # Daily oil prices
│   ├── holidays_events.csv                    # Holiday & event calendar
│   ├── transactions.csv                       # Daily transaction counts
│   ├── sample_submission.csv                  # Submission format template
│   └── submission.csv                         # Generated predictions
│
└── ...
```

### Notebooks Description

| Notebook | Purpose |
|:---------|:--------|
| `store_sales_complete_analysis.ipynb` | **Full analysis** — Detailed EDA, feature engineering, model training, and evaluation with rich markdown explanations |
| `Notebook/store-sales-time-series-f.ipynb` | **Kaggle submission** — Optimized notebook for Kaggle kernel execution with competition submission pipeline |

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.10+** installed along with the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost
```

### Running the Analysis

1. **Clone or download** this repository
2. **Navigate** to the project directory
3. **Open** the notebook in Jupyter:
   ```bash
   jupyter notebook store_sales_complete_analysis.ipynb
   ```
4. **Run all cells** sequentially to reproduce the full analysis

### Running on Kaggle

1. Upload `Notebook/store-sales-time-series-f.ipynb` to Kaggle
2. Attach the competition dataset
3. Run the notebook to generate `submission.csv`

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|:-----------|:--------|:--------|
| **Python** | 3.12 | Core programming language |
| **NumPy** | Latest | Numerical computing |
| **Pandas** | Latest | Data manipulation & analysis |
| **Matplotlib** | Latest | Static visualizations |
| **Seaborn** | Latest | Statistical data visualization |
| **scikit-learn** | Latest | ML utilities, Random Forest, preprocessing |
| **LightGBM** | Latest | Gradient boosting (primary model) |
| **XGBoost** | Latest | Gradient boosting (secondary model) |
| **Jupyter Notebook** | Latest | Interactive development environment |

---

## 👨‍💻 Author

**Abhishek**

- 📧 Contact: Available upon request
- 🔗 Kaggle: [Profile](https://www.kaggle.com)

---

<div align="center">

### ⭐ If you found this project helpful, please consider giving it a star!

*Built with ❤️ for the Kaggle community*

</div>
