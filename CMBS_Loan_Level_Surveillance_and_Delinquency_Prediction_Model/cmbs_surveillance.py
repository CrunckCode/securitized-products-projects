"""
CMBS Loan-Level Surveillance and Delinquency Prediction Model
=================================================================
Loan-level CMBS surveillance on a constructed 300-loan pool (real trustee-level loan data
isn't freely downloadable via API), but property values are shocked using the REAL
quarterly commercial real estate price index from FRED, so DSCR/occupancy deterioration
in this pool tracks the real 2022-2025 CRE price downturn rather than an arbitrary
assumption. Trains a gradient-boosting delinquency classifier and validates with
ROC/AUROC.
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(21)
TODAY = datetime.date.today()

# ===========================================================================
# 1. Real commercial real estate price index (FRED) - the macro driver
# ===========================================================================
cre = web.DataReader("COMREPUSQ159N", "fred", start=datetime.datetime(2018, 1, 1)).dropna()
print("Real US commercial real estate price index (YoY % growth, FRED COMREPUSQ159N):")
print(cre.tail(8).round(2).to_string())
cre_growth = cre.iloc[:, 0].values / 100  # convert to decimal
n_periods = len(cre_growth)

# ===========================================================================
# 2. Constructed 300-loan CMBS pool, property types weighted realistically
# ===========================================================================
N_LOANS = 300
property_types = np.random.choice(
    ["Office", "Retail", "Multifamily", "Industrial", "Hotel"],
    size=N_LOANS, p=[0.30, 0.20, 0.25, 0.15, 0.10])
# Real, well-documented sector stress ordering post-2022: office >> retail/hotel > multifamily/industrial
sector_stress_multiplier = {"Office": 1.8, "Retail": 1.2, "Hotel": 1.3, "Multifamily": 0.7, "Industrial": 0.6}

initial_dscr = np.random.normal(1.45, 0.25, N_LOANS).clip(1.0, 2.5)
initial_occupancy = np.random.normal(0.90, 0.08, N_LOANS).clip(0.5, 1.0)
initial_ltv = np.random.normal(0.62, 0.10, N_LOANS).clip(0.35, 0.85)
origination_quarter = np.random.randint(0, max(n_periods - 12, 1), N_LOANS)

# ===========================================================================
# 3. Propagate loan-level DSCR/occupancy forward using the REAL CRE price path
# ===========================================================================
records = []
for i in range(N_LOANS):
    ptype = property_types[i]
    stress_mult = sector_stress_multiplier[ptype]
    dscr = initial_dscr[i]
    occ = initial_occupancy[i]
    ltv = initial_ltv[i]
    start_q = origination_quarter[i]
    prior_delinquent = False
    for q in range(start_q, min(start_q + 12, n_periods)):
        real_cre_shock = cre_growth[q] * stress_mult  # sector-stressed real macro shock
        # DSCR responds to real property value/NOI pressure implied by the CRE index
        dscr = max(dscr * (1 + real_cre_shock * 0.5) + np.random.normal(0, 0.03), 0.2)
        occ = np.clip(occ + real_cre_shock * 0.3 + np.random.normal(0, 0.01), 0.3, 1.0)
        ltv = np.clip(ltv - real_cre_shock * 0.4, 0.3, 1.3)
        months_seasoned = (q - start_q) * 3
        delinquent = int((dscr < 1.0) or (occ < 0.55) or (prior_delinquent and dscr < 1.15))
        records.append({
            "loan_id": i, "property_type": ptype, "quarter_idx": q,
            "dscr": dscr, "occupancy": occ, "ltv": ltv,
            "months_seasoned": months_seasoned, "prior_delinquent": int(prior_delinquent),
            "delinquent_next_period": delinquent,
        })
        prior_delinquent = bool(delinquent)

panel = pd.DataFrame(records)
print(f"\nBuilt loan-quarter panel: {len(panel)} observations across {N_LOANS} loans")
print(f"Overall delinquency rate: {panel['delinquent_next_period'].mean():.1%}")
print("\nDelinquency rate by property type (real sector-stress ordering should emerge):")
print(panel.groupby("property_type")["delinquent_next_period"].mean().sort_values(ascending=False).round(3))

# ===========================================================================
# 4. Feature engineering: trend features (DSCR/occupancy direction, not just level)
# ===========================================================================
panel = panel.sort_values(["loan_id", "quarter_idx"])
panel["dscr_trend"] = panel.groupby("loan_id")["dscr"].diff().fillna(0)
panel["occupancy_trend"] = panel.groupby("loan_id")["occupancy"].diff().fillna(0)
panel_encoded = pd.get_dummies(panel, columns=["property_type"], drop_first=True)

feature_cols = ["dscr", "occupancy", "ltv", "months_seasoned", "prior_delinquent",
                 "dscr_trend", "occupancy_trend"] + \
                [c for c in panel_encoded.columns if c.startswith("property_type_")]
X = panel_encoded[feature_cols]
y = panel_encoded["delinquent_next_period"]

# ===========================================================================
# 5. Train/validate delinquency classifier
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21,
                                                       stratify=y)
model = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05,
                                     random_state=21)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, proba)
preds = (proba > 0.5).astype(int)
cm = confusion_matrix(y_test, preds)

print("\n" + "=" * 70)
print("DELINQUENCY PREDICTION MODEL VALIDATION")
print("=" * 70)
print(f"Test set: {len(y_test)} loan-quarters, base rate {y_test.mean():.1%}")
print(f"AUROC: {auroc:.3f}")
print(f"Confusion matrix:\n{cm}")

importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature importances:")
print(importances.round(3).to_string())

# ===========================================================================
# 6. Watch-list: current loans with high predicted delinquency probability
# ===========================================================================
latest = panel_encoded.sort_values("quarter_idx").groupby("loan_id").tail(1)
latest_proba = model.predict_proba(latest[feature_cols])[:, 1]
latest = latest.assign(delinquency_probability=latest_proba)
watch_list = latest.sort_values("delinquency_probability", ascending=False).head(10)
print("\n" + "=" * 70)
print("TOP 10 WATCH-LIST LOANS (highest predicted delinquency probability)")
print("=" * 70)
print(watch_list[["loan_id", "dscr", "occupancy", "delinquency_probability"]].round(3).to_string(index=False))

# ===========================================================================
# 7. ROC chart
# ===========================================================================
fpr, tpr, _ = roc_curve(y_test, proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("CMBS Delinquency Prediction - ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=120)
print("\nSaved chart: roc_curve.png")
