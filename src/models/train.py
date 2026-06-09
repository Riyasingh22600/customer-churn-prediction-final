import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from src.data.load_data import load_data
from src.data.preprocess import split_data
from src.models.utils import save_object


TOP_N_FEATURES = 15  # you can tune this


def train_and_serialize(path: str | None = None):
    df = load_data(path)

    # Drop raw target column if present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Split
    X_train, X_test, y_train, y_test = split_data(df, target="Churn_Encoded")

    # Train baseline model
    baseline_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    baseline_clf.fit(X_train, y_train)

    # Rank features
    importances = baseline_clf.feature_importances_
    feat_names = X_train.columns
    sorted_feats = sorted(zip(feat_names, importances), key=lambda x: -x[1])

    # Select top N
    top_features = [f for f, _ in sorted_feats[:TOP_N_FEATURES]]
    print(f"Selected Top {TOP_N_FEATURES} Features:", top_features)

    # Retrain on selected features only
    X_train_sel = X_train[top_features]
    X_test_sel = X_test[top_features]

    clf = RandomForestClassifier(
        n_estimators=300,  # slightly stronger final model
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    clf.fit(X_train_sel, y_train)

    preds = clf.predict(X_test_sel)
    proba = clf.predict_proba(X_test_sel)[:, 1]

    print(classification_report(y_test, preds))
    print("AUC:", roc_auc_score(y_test, proba))

    # Save model and selected features
    save_object(clf, "model_rf.joblib")
    save_object(top_features, "selected_features.joblib")
    print("✅ Model + feature list saved in models/")

if __name__ == '__main__':
    train_and_serialize()