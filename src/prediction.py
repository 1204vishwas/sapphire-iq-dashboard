"""
Return Risk Prediction Classification Module
Amazon India Sales Analytics & AI-Powered Business Intelligence System
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

from src.feature_engineering import prepare_classification_data


def compare_classification_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame, Any]:
    """
    Train and compare multiple classifiers on return prediction:
    - Logistic Regression (Balanced)
    - Decision Tree Classifier
    - Random Forest Classifier
    - XGBoost Classifier

    Returns evaluation summary, leaderboard dataframe, and the best model.
    """
    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    candidate_classifiers = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss")
    }

    results = []
    trained_models = {}

    for name, clf in candidate_classifiers.items():
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else preds
        trained_models[name] = clf

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        roc = float(roc_auc_score(y_test, probs))

        results.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(roc, 4)
        })

    leaderboard = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    best_model_name = leaderboard.iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    print("=== Return Risk Classification Leaderboard ===")
    print(leaderboard.to_string(index=False))
    print(f"\nBest Classifier Selected: {best_model_name}")

    eval_summary = {
        "leaderboard": leaderboard,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "y_test": y_test,
        "predictions": {name: m.predict(X_test) for name, m in trained_models.items()},
        "probabilities": {name: (m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else m.predict(X_test)) for name, m in trained_models.items()},
        "confusion_matrix": confusion_matrix(y_test, best_model.predict(X_test))
    }

    return eval_summary, leaderboard, best_model


def train_and_save_best_classifier(
    df: pd.DataFrame,
    model_path: str = "models/return_prediction_model.pkl"
) -> Dict[str, Any]:
    """
    Run model comparison on order return risk, select the winner, refit and save artifact.
    """
    X_train, X_test, y_train, y_test, feature_names = prepare_classification_data(df)

    eval_summary, leaderboard, best_candidate = compare_classification_models(
        X_train, X_test, y_train, y_test
    )

    best_model_name = eval_summary["best_model_name"]
    pos_weight = (len(y_train) - sum(y_train)) / max(sum(y_train), 1)

    # Refit chosen model on full feature space
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    if best_model_name == "Logistic Regression":
        final_clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    elif best_model_name == "Decision Tree":
        final_clf = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    elif best_model_name == "Random Forest":
        final_clf = RandomForestClassifier(n_estimators=120, max_depth=6, class_weight="balanced", random_state=42)
    else:
        final_clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, scale_pos_weight=pos_weight, random_state=42, eval_metric="logloss")

    final_clf.fit(X_full, y_full)

    # Feature Importance (if applicable)
    feat_importances = {}
    if hasattr(final_clf, "feature_importances_"):
        for f, imp in zip(feature_names, final_clf.feature_importances_):
            feat_importances[f] = float(imp)
    elif hasattr(final_clf, "coef_"):
        for f, imp in zip(feature_names, final_clf.coef_[0]):
            feat_importances[f] = float(abs(imp))

    artifact = {
        "model": final_clf,
        "model_name": best_model_name,
        "feature_names": feature_names,
        "feature_importances": dict(sorted(feat_importances.items(), key=lambda x: x[1], reverse=True)[:15]),
        "leaderboard": leaderboard,
        "categories": sorted(df["Category"].dropna().unique().tolist()),
        "products": sorted(df["Product"].dropna().unique().tolist()),
        "payment_methods": sorted(df["Payment_Method"].dropna().unique().tolist()),
        "fulfillments": sorted(df["Fulfillment"].dropna().unique().tolist()),
        "states": sorted(df["Ship_State"].dropna().unique().tolist())
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(artifact, model_path)
    print(f"Saved best return prediction model ({best_model_name}) to: {model_path}")

    return artifact


def predict_order_return_risk(
    order_data: Dict[str, Any],
    artifact: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Given a single order dictionary, compute return probability and risk level.
    """
    model = artifact["model"]
    feature_names = artifact["feature_names"]

    # Compute total sales if not directly provided
    qty = int(order_data.get("Quantity", 1))
    price = float(order_data.get("Unit_Price_INR", 1000.0))
    disc = float(order_data.get("Discount_Pct", 0.0))
    total_sales = float(order_data.get("Total_Sales_INR", qty * price * (1.0 - disc)))

    input_df = pd.DataFrame([{
        "Category": order_data.get("Category", "Electronics & Mobiles"),
        "Product": order_data.get("Product", "Wireless Earbuds"),
        "Payment_Method": order_data.get("Payment_Method", "UPI"),
        "Fulfillment": order_data.get("Fulfillment", "Amazon (FBA)"),
        "Ship_State": order_data.get("Ship_State", "Maharashtra"),
        "Quantity": qty,
        "Unit_Price_INR": price,
        "Discount_Pct": disc,
        "Total_Sales_INR": total_sales
    }])

    # One-hot encode with matching columns
    encoded_df = pd.get_dummies(input_df)
    encoded_aligned = encoded_df.reindex(columns=feature_names, fill_value=0)

    prob = float(model.predict_proba(encoded_aligned)[0, 1])

    if prob >= 0.65:
        risk_level = "High Risk"
        badge_color = "red"
        recommendation = "High risk of return! Consider extra packaging checks, pre-dispatch customer verification, or courier tracking."
    elif prob >= 0.40:
        risk_level = "Medium Risk"
        badge_color = "orange"
        recommendation = "Moderate risk. Standard delivery with delivery notification SMS is advised."
    else:
        risk_level = "Low Risk"
        badge_color = "green"
        recommendation = "Low return likelihood. Fast-track order for fulfillment."

    return {
        "return_probability": round(prob * 100, 2),
        "risk_level": risk_level,
        "badge_color": badge_color,
        "recommendation": recommendation
    }
