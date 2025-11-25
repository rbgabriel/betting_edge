# /agent_modules/prediction_agent_wrapper.py (NEW/UPDATED FILE)

import pickle
import json
from xgboost import XGBClassifier
import pandas as pd
import numpy as np

class PredictionAgentLC: # Renamed for consistency, but acts as the wrapper for XGBoost
    def __init__(self, model_path="xgb_model.json", mappings_path="team_mappings.pkl"):
        self.model = self._load_model(model_path)
        self.home_encoder, self.away_encoder = self._load_encoders(mappings_path)
        self.target_names = ["home_win", "away_win", "draw"] # Based on prepare_features in train_xgboost.py

    def _load_model(self, path):
        """Loads the XGBoost model."""
        model = XGBClassifier()
        model.load_model(path)
        return model

    def _load_encoders(self, path):
        """Loads the LabelEncoders."""
        with open(path, "rb") as f:
            mappings = pickle.load(f)
        return mappings["home_encoder"], mappings["away_encoder"]

    def _preprocess_match_data(self, match_details: dict) -> np.ndarray:
        """
        Preprocesses a single match's details into features for the XGBoost model.
        
        Args:
            match_details (dict): A dictionary containing match details, 
                                  expected to have 'teams' -> 'home' -> 'name' and 'away' -> 'name'.
        
        Returns:
            np.ndarray: A 1x2 NumPy array of encoded home and away team IDs.
        """
        home_team_name = match_details['teams']['home']['name']
        away_team_name = match_details['teams']['away']['name']

        # Handle unknown teams gracefully by using a consistent value, e.g., -1 or the length of known classes
        # This is a basic approach; a more robust solution might retrain or use an 'unknown' category.
        try:
            home_team_enc = self.home_encoder.transform([home_team_name])[0]
        except ValueError:
            home_team_enc = -1 # Or some other indicator for unknown
            print(f"Warning: Home team '{home_team_name}' not in encoder vocabulary.")

        try:
            away_team_enc = self.away_encoder.transform([away_team_name])[0]
        except ValueError:
            away_team_enc = -1 # Or some other indicator for unknown
            print(f"Warning: Away team '{away_team_name}' not in encoder vocabulary.")

        # The model expects a 2D array, even for a single prediction
        return np.array([[home_team_enc, away_team_enc]])

    def invoke(self, match_details: dict) -> dict:
        """
        Predicts the outcome probabilities for a given match.

        Args:
            match_details (dict): Dictionary containing match information, 
                                  e.g., from the pipeline's filtered_matches.

        Returns:
            dict: A dictionary with predicted winner and probabilities for home win, away win, and draw.
        """
        if not self.model or not self.home_encoder or not self.away_encoder:
            return {"status": "error", "message": "Prediction model or encoders not loaded."}

        # Preprocess the input match details
        features = self._preprocess_match_data(match_details)

        # Make prediction
        # predict_proba returns probabilities for each class [home_win, away_win, draw]
        probabilities = self.model.predict_proba(features)[0] 
        
        home_win_prob = probabilities[0]
        away_win_prob = probabilities[1]
        draw_prob = probabilities[2]

        # Determine the predicted winner
        predicted_outcome_index = np.argmax(probabilities)
        predicted_winner_model = self.target_names[predicted_outcome_index]

        # Map to more readable names if desired
        if predicted_winner_model == "home_win":
            predicted_winner_name = match_details['teams']['home']['name']
        elif predicted_winner_model == "away_win":
            predicted_winner_name = match_details['teams']['away']['name']
        else: # "draw"
            predicted_winner_name = "Draw"

        return {
            "status": "ok",
            "predicted_winner_model": predicted_winner_name, # The actual team name or "Draw"
            "home_win_probability": home_win_prob,
            "away_win_probability": away_win_prob,
            "draw_probability": draw_prob,
            "raw_probabilities": probabilities.tolist()
        }