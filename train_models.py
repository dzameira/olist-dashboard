import os
import json
import joblib
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import TargetEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_PATH = "olist_master_featured_clean.parquet"
MODELS_DIR = "models"

LEAKAGE_AND_ID_COLS = [
    "is_late",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "seller_handling_days",
    "order_id",
    "customer_id",
    "customer_unique_id",
    "primary_seller_id",
    "order_status",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
    "latest_shipping_limit_date",
    "seller_weather_start_date",
    "seller_weather_end_date",
    "customer_weather_start_date",
    "customer_weather_end_date",
    "seller_city",
    "customer_city",
]

print("STEP 1: Load frozen dataset & build leak-free train/test split")
master_df = pd.read_parquet(DATA_PATH)

train_df, test_df = train_test_split(
    master_df, test_size=0.20, random_state=42, stratify=master_df["is_late"]
)

# Split-safe seller volume (computed strictly on train_df)
seller_volume_map = train_df.groupby("primary_seller_id")["order_id"].nunique()
train_df["seller_order_count"] = train_df["primary_seller_id"].map(seller_volume_map).fillna(1).astype(np.int32)
test_df["seller_order_count"] = test_df["primary_seller_id"].map(seller_volume_map).fillna(1).astype(np.int32)

train_df["is_top_seller"] = (train_df["seller_order_count"] >= 100).astype(np.int8)
test_df["is_top_seller"] = (test_df["seller_order_count"] >= 100).astype(np.int8)

# Split-safe Out-of-Fold Target Encoding
te_cols = ["primary_seller_id", "customer_state", "primary_category"]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
target_encoder = TargetEncoder(cv=skf, smooth="auto")

train_te = target_encoder.fit_transform(train_df[te_cols], train_df["is_late"])
test_te = target_encoder.transform(test_df[te_cols])

for i, col in enumerate(te_cols):
    train_df[f"{col}_hist_late_rate"] = train_te[:, i].astype(np.float32)
    test_df[f"{col}_hist_late_rate"] = test_te[:, i].astype(np.float32)

# Drop target, leakage, and ID columns
cols_to_drop = [c for c in LEAKAGE_AND_ID_COLS if c in train_df.columns]
X_train = train_df.drop(columns=cols_to_drop)
y_train = train_df["is_late"].astype(int)

X_test = test_df.drop(columns=cols_to_drop)
y_test = test_df["is_late"].astype(int)

cat_cols = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_test[col] = X_test[col].astype("category")

print(f"  X_train: {X_train.shape} | X_test: {X_test.shape} | Test late rate: {y_test.mean():.2%}")

print("\nSTEP 2: Train & compare 3 distinct model families")
scale_weight = float((y_train == 0).sum() / (y_train == 1).sum())

lr_pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
])

models = {
    "Logistic Regression": lr_pipeline,
    "LightGBM": LGBMClassifier(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=scale_weight,
        enable_categorical=True,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    ),
}

results, fitted = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # Find optimal F1 threshold for each model
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    f1s = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)
    best_idx = int(np.argmax(f1s))
    opt_thresh = float(thresholds[best_idx])
    preds_opt = (proba >= opt_thresh).astype(int)

    results[name] = {
        "ROC_AUC": float(roc_auc_score(y_test, proba)),
        "PR_AUC": float(average_precision_score(y_test, proba)),
        "Optimal_Threshold": opt_thresh,
        "Precision_at_Opt": float(precision_score(y_test, preds_opt)),
        "Recall_at_Opt": float(recall_score(y_test, preds_opt)),
        "F1_at_Opt": float(f1_score(y_test, preds_opt)),
    }
    fitted[name] = model
    r = results[name]
    print(
        f"  {name:20s} ROC-AUC={r['ROC_AUC']:.4f} | PR-AUC={r['PR_AUC']:.4f} | "
        f"F1={r['F1_at_Opt']:.4f} (P={r['Precision_at_Opt']:.4f}, R={r['Recall_at_Opt']:.4f} @ t={opt_thresh:.2f})"
    )

# Select best model by PR-AUC (primary metric for ~6.85% minority class)
best_name = max(results, key=lambda k: results[k]["PR_AUC"])
best_model = fitted[best_name]
print(f"\nSTEP 3: Best model by test PR-AUC = {best_name} ({results[best_name]['PR_AUC']:.4f})")

print("\nSTEP 4: Global SHAP on 1,000 stratified test rows")
sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
if best_name in ("XGBoost", "LightGBM"):
    explainer = shap.TreeExplainer(best_model)
    raw_shap = explainer.shap_values(sample)
    # Handle LightGBM returning a list [class_0, class_1] or 3D array
    if isinstance(raw_shap, list):
        global_shap_values = raw_shap[1]
    elif raw_shap.ndim == 3:
        global_shap_values = raw_shap[:, :, 1]
    else:
        global_shap_values = raw_shap
else:
    # Fallback if Logistic Regression ever wins
    bg = X_train.sample(200, random_state=42)
    explainer = shap.Explainer(best_model.predict_proba, bg)
    global_shap_values = explainer(sample).values[:, :, 1]

print("\nSTEP 5: Save version-proof artifacts to models/")
os.makedirs(MODELS_DIR, exist_ok=True)
for f in os.listdir(MODELS_DIR):
    if f.endswith((".joblib", ".json", ".npy", ".parquet", ".csv")):
        os.remove(os.path.join(MODELS_DIR, f))

if isinstance(best_model, XGBClassifier):
    best_model.save_model(f"{MODELS_DIR}/best_model.json")
    model_kind = "xgboost_json"
else:
    joblib.dump(best_model, f"{MODELS_DIR}/best_model.joblib", compress=3)
    model_kind = "joblib"

# Save schema & lookup maps needed for Streamlit real-time inference
schema = {
    "feature_columns": X_train.columns.tolist(),
    "numeric_columns": num_cols,
    "categorical_columns": cat_cols,
    "category_choices": {col: sorted(X_train[col].cat.categories.tolist()) for col in cat_cols},
    "numeric_medians": {col: float(X_train[col].median()) for col in num_cols},
    "target_encoded_columns": te_cols,
    "global_prior_late_rate": float(y_train.mean()),
    "optimal_threshold": results[best_name]["Optimal_Threshold"],
}

# Save lookup tables for Streamlit inference (seller volume + target encoder)
joblib.dump(
    {
        "seller_volume_map": seller_volume_map.to_dict(),
        "target_encoder": target_encoder,
    },
    f"{MODELS_DIR}/inference_lookups.joblib",
    compress=3,
)

sample.to_parquet(f"{MODELS_DIR}/shap_sample.parquet", index=False)
np.save(f"{MODELS_DIR}/global_shap_values.npy", global_shap_values)

with open(f"{MODELS_DIR}/schema.json", "w") as f:
    json.dump(schema, f, indent=2)

with open(f"{MODELS_DIR}/results.json", "w") as f:
    json.dump({"best_model": best_name, "model_kind": model_kind, "metrics": results}, f, indent=2)

print("  Saved all artifacts to models/. dashboard development coming")