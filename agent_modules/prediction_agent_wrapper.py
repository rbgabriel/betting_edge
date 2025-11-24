# agent_modules/prediction_agent_wrapper.py

import pickle
import xgboost as xgb
import numpy as np

class PredictionAgentLC:
    """
    Prediction Agent using the trained XGBoost model.
    Uses ONLY 2 features:
        - home_team_id_encoded
        - away_team_id_encoded
    """

    def __init__(self):

        # Load your XGBoost model
        try:
            self.model = xgb.XGBClassifier()
            self.model.load_model("xgb_model.json")
        except Exception as e:
            print("❌ Could not load XGBoost model:", e)
            self.model = None

        # Load team encoders
        try:
            with open("team_mappings.pkl", "rb") as f:
                enc = pickle.load(f)
            self.home_encoder = enc["home_encoder"]
            self.away_encoder = enc["away_encoder"]
        except Exception as e:
            print("❌ Could not load encoders:", e)
            self.home_encoder = None
            self.away_encoder = None

    def encode_team(self, name, encoder):
        """Safely encodes team names, unknown teams get 0"""
        try:
            return encoder.transform([name])[0]
        except:
            return 0  # fallback for unseen teams

    def invoke(self, match):
        """
        Produces probabilities:
        - home win
        - draw
        - away win
        """

        if self.model is None:
            return {"status": "error", "message": "Model not loaded"}

        try:
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]

            # Encode BOTH teams
            h_enc = self.encode_team(home, self.home_encoder)
            a_enc = self.encode_team(away, self.away_encoder)

            # XGBoost expects shape (1, 2)
            X = np.array([[h_enc, a_enc]])

            # Predict class probabilities
            probs = self.model.predict_proba(X)[0]

            # probs comes as array([p_home, p_draw, p_away])
            result = {
                "status": "ok",
                "home_team": home,
                "away_team": away,
                "home_win_probability": float(round(probs[0], 3)),
                "draw_probability": float(round(probs[1], 3)),
                "away_win_probability": float(round(probs[2], 3)),
                "winner": home if probs[0] > probs[2] else away
            }

            return result

        except Exception as e:
            return {"status": "error", "message": str(e)}
