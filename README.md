# datascore

ML readiness scoring for tabular datasets.

Point it at a DataFrame and get a structured report telling you whether your data is ready for ML training — and if not, exactly why.

## Install

pip install datascore

## Usage

from datascore import score

report = score(df, target="churn")
report.show()

## Output

datascore Report
========================================
Rows: 7043 | Features: 21 | Target: Churn
Score: 85/100 — READY

WARNINGS
  - High cardinality: customerID has 7043 unique values
  - High cardinality: TotalCharges has 6531 unique values
  - High skew in SeniorCitizen: 1.8332

INFO
  - No constant features detected
  - No infinite values detected
  - Class balance: 73/27

## What it checks

| Category | Checks |
|---|---|
| Completeness | Missing values, high missing rate per column |
| Integrity | Duplicate rows, constant features, infinite values |
| ML Readiness | Class imbalance, target leakage risk, high cardinality |
| Distribution | Skew, outliers per column |

## Scoring

Starts at 100. Each blocker deducts 15 points, each warning deducts 5.

| Score | Verdict |
|---|---|
| 80-100 | READY |
| 50-79 | NEEDS WORK |
| 0-49 | NOT READY |

## Why not Great Expectations or Pandera?

Those tools validate data against rules you define. datascore tells you what the problems are without you having to know what to look for first. Assessment, not validation.