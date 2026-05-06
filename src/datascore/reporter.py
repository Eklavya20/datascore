from dataclasses import dataclass, field


@dataclass
class Report:
    score: int
    verdict: str
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def show(self):
        print("\ndatascore Report")
        print("=" * 40)
        rows, cols = self.raw["shape"]
        target = self.raw["target"] or "not specified"
        print(f"Rows: {rows} | Features: {cols} | Target: {target}")
        print(f"\nScore: {self.score}/100 — {self.verdict}\n")

        if self.blockers:
            print("BLOCKERS")
            for b in self.blockers:
                print(f"  - {b}")

        if self.warnings:
            print("\nWARNINGS")
            for w in self.warnings:
                print(f"  - {w}")

        if self.info:
            print("\nINFO")
            for i in self.info:
                print(f"  - {i}")

        print()


def build_report(results: dict) -> Report:
    blockers = []
    warnings = []
    info = []

    c = results["completeness"]
    i = results["integrity"]
    ml = results["ml_readiness"]
    d = results["distribution"]

    # --- Completeness ---
    if c["total_missing_rate"] > 0.2:
        blockers.append(f"High overall missing rate: {c['total_missing_rate']*100:.1f}%")
    elif c["total_missing_rate"] > 0:
        warnings.append(f"Missing values detected: {c['total_missing_rate']*100:.1f}% overall")

    for col, rate in c["high_missing_cols"].items():
        blockers.append(f"{col}: {rate*100:.1f}% missing values")

    # --- Integrity ---
    if i["duplicate_rows"] > 0:
        warnings.append(f"{i['duplicate_rows']} duplicate rows detected")

    if i["constant_cols"]:
        blockers.append(f"Constant features (zero variance): {', '.join(i['constant_cols'])}")

    if i["near_constant_cols"]:
        warnings.append(f"Near-constant features: {', '.join(i['near_constant_cols'])}")

    if i["infinite_values"] > 0:
        blockers.append(f"{i['infinite_values']} infinite values detected")

    # --- ML Readiness ---
    if ml.get("class_imbalanced"):
        minority = ml["class_balance_minority"]
        warnings.append(f"Class imbalance: minority class is {minority*100:.1f}% of data")

    for leak in ml.get("leakage_risk_cols", []):
        blockers.append(
            f"Leakage risk: {leak['feature']} correlates {leak['correlation']} with target"
        )

    for col, cardinality in ml.get("high_cardinality_cols", {}).items():
        warnings.append(f"High cardinality: {col} has {cardinality} unique values")

    # --- Distribution ---
    for col, skew in d["skewed_cols"].items():
        warnings.append(f"High skew in {col}: {skew}")

    for col, count in d["outlier_cols"].items():
        info.append(f"Outliers in {col}: {count} rows")

    # --- Info ---
    if not i["constant_cols"]:
        info.append("No constant features detected")
    if i["infinite_values"] == 0:
        info.append("No infinite values detected")
    if ml.get("class_balance_minority") is not None:
        minority = ml["class_balance_minority"]
        info.append(f"Class balance: {(1-minority)*100:.0f}/{minority*100:.0f}")

    # --- Score ---
    score = 100
    score -= len(blockers) * 15
    score -= len(warnings) * 5
    score = max(0, score)

    if score >= 80:
        verdict = "READY"
    elif score >= 50:
        verdict = "NEEDS WORK"
    else:
        verdict = "NOT READY"

    return Report(
        score=score,
        verdict=verdict,
        blockers=blockers,
        warnings=warnings,
        info=info,
        raw=results,
    )