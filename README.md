# datascore

![CI](https://github.com/Eklavya20/datascore/actions/workflows/ci.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/datascore)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

ML readiness scoring for tabular datasets.

Point it at a DataFrame and get a structured report telling you whether your data is ready for ML training — and if not, exactly why and in what order to fix it.

## Install

```bash
pip install datascore
```

## Usage

```python
import seaborn as sns
from datascore import score

df = sns.load_dataset("titanic")
report = score(df, target="survived")
report.show()
```

## Output

```text
datascore Report

Rows: 891 | Features: 15 | Target: survived
Score: 45/100 — NOT READY

BLOCKERS

- age: 19.9% missing values
- deck: 77.2% missing values

WARNINGS

- Missing values detected: 6.5% overall
- 107 duplicate rows detected
- High skew in sibsp: 3.6891
- High skew in parch: 2.7445
- High skew in fare: 4.7793

INFO

- Outliers in age: 11 rows
- Outliers in sibsp: 46 rows
- Outliers in parch: 213 rows
- Outliers in fare: 116 rows
- No constant features detected
- No infinite values detected
- Class balance: 62/38
```

## Save report to markdown

```python
report.save("report.md")
```

## What it checks

| Category | Checks |
|----------|----------|
| Completeness | Missing values, high missing rate per column (>5%) |
| Integrity | Duplicate rows, constant features, infinite values |
| ML Readiness | Class imbalance, target leakage risk, high cardinality categoricals |
| Distribution | Skew per numerical column, outliers via IQR |

## Scoring

Starts at 100. Each blocker deducts 15 points, each warning deducts 5.

| Score | Verdict |
|---------|---------|
| 80–100 | READY |
| 50–79 | NEEDS WORK |
| 0–49 | NOT READY |

## Why not Great Expectations or Pandera?

Those tools validate data against rules you define upfront.

**datascore** requires no configuration — it tells you what the problems are without you having to know what to look for first.

Assessment, not validation.

## License

MIT