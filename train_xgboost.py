import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import pickle
import json

DB_PATH = "betting_edge.db"

# ---------------------------------------------------------
# 1. LOAD MATCHES FROM DB
# ---------------------------------------------------------
def load_finished_matches():
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT 
            match_id,
            home_team_name,
            away_team_name,
            home_score,
            away_score,
            status
        FROM matches
        WHERE LOWER(status) IN (
            'finished', 'completed', 'ft', 'match finished'
        )
        AND home_score IS NOT NULL
        AND away_score IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
def prepare_features(df):
    # Label encode teams
    home_encoder = LabelEncoder()
    away_encoder = LabelEncoder()

    df["home_team_enc"] = home_encoder.fit_transform(df["home_team_name"])
    df["away_team_enc"] = away_encoder.fit_transform(df["away_team_name"])

    # Outcome label
    def outcome(row):
        if row["home_score"] > row["away_score"]:
            return 0  # home win
        elif row["home_score"] < row["away_score"]:
            return 1  # away win
        else:
            return 2  # draw

    df["label"] = df.apply(outcome, axis=1)

    X = df[["home_team_enc", "away_team_enc"]].values
    y = df["label"].values

    return X, y, home_encoder, away_encoder


# ---------------------------------------------------------
# 3. TRAIN XGBOOST
# ---------------------------------------------------------
def train_xgboost(X, y):
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.06,
        max_depth=5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    model.fit(X, y)
    return model


# ---------------------------------------------------------
# 4. SAVE MODEL + ENCODERS
# ---------------------------------------------------------
def save_artifacts(model, home_encoder, away_encoder):
    model.save_model("xgb_model.json")
    with open("team_mappings.pkl", "wb") as f:
        pickle.dump({"home_encoder": home_encoder, "away_encoder": away_encoder}, f)


# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------
def main():
    print("📥 Loading finished matches from DB...")
    df = load_finished_matches()

    if df.empty:
        print("❌ No finished matches found. Run your app first to populate DB.")
        return

    print(f"✅ Loaded {len(df)} matches.")

    # Prepare training data
    X, y, home_encoder, away_encoder = prepare_features(df)

    print("⚙️ Training XGBoost model ...")
    model = train_xgboost(X, y)

    print("💾 Saving model and encoders ...")
    save_artifacts(model, home_encoder, away_encoder)

    print("✅ Model saved as xgb_model.json")
    print("✅ Encoders saved as team_mappings.pkl")

     # --------------------------
    # Evaluation on Training Data
    # --------------------------
    raw_preds = model.predict(X)

    # If model outputs class labels (1D)
    if len(raw_preds.shape) == 1:
        preds = raw_preds.astype(int)
    else:
        preds = np.argmax(raw_preds, axis=1)

    print("\n📊 Training Accuracy:", round(accuracy_score(y, preds), 3))
    print("\n📄 Classification Report:")
    print(classification_report(
        y, preds,
        target_names=["home_win", "away_win", "draw"]
    ))

