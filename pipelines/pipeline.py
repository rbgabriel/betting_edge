# /pipelines/pipeline.py (DEFINITIVE FIX)

import os
import json
from typing import Dict, Any, List

# No direct Streamlit imports here!

# Import functions/classes from their respective modules
from query_agent import parse_user_query 
from agent_modules.prediction_agent_wrapper import PredictionAgentLC
from agent_modules.verification_agent_wrapper import VerificationAgentLC # Assuming this is the correct class name inside the wrapper
from agent_modules.behavior_agent_wrapper import BehaviorAgentLC
from agent_modules.recommendation_agent_wrapper import RecommendationAgentLC
from agent_modules.ethics_agent_wrapper import EthicsAgentLC

# Import from utils.py. Note that init_data_agent, fetch_matches_from_db, etc.
# are now in utils.py and should be imported from there.
from utils import (
    fetch_matches_from_db,
    init_data_agent,
    fetch_match_stats, # Need to be imported here for run_deep_analysis
    fetch_odds,       # Need to be imported here for run_deep_analysis
    get_db_connection# We will call this to ensure DataAgent is set up, but the fetch is direct.
)

class BettingEdgePipeline:
    def __init__(self):
        # The query_agent is just a function, not a class instance itself
        self.query_agent_func = parse_user_query 
        self.prediction_agent = PredictionAgentLC()
        self.verification_agent = VerificationAgentLC()
        self.behavior_agent = BehaviorAgentLC()
        self.recommendation_agent = RecommendationAgentLC()
        self.ethics_agent = EthicsAgentLC()

    def run(self, user_query: str):
        # No st.info calls here!
        parsed_query_obj = self.query_agent_func(user_query) # Use the function directly

        query_dict = parsed_query_obj.dict()
        
        sport_type = query_dict.get("sport_type")
        team_name = query_dict.get("team_name")
        competition_code = query_dict.get("competition_code")
        season = query_dict.get("season") # This can be None or an int

        if not sport_type:
            return {"status": "query_error", "message": "Could not determine sport type from your query."}

        # Ensure the DataAgent is initialized in the Streamlit session for other modules that might need it
        # This function (init_data_agent) *is* allowed to use st.session_state because it's in utils.py,
        # which is imported by streamlit_app.py. However, no direct st. calls in here.
        init_data_agent(sport_type) # This is a helper, not a display component.

        if sport_type == "football":
            all_matches_df = fetch_matches_from_db(
                sport_type=sport_type,
                league_name=competition_code,
                team_name=team_name,
                season=season # Pass the season parameter
            )
        elif sport_type in ["college_football", "basketball"]:
            all_matches_df = fetch_matches_from_db(
                sport_type=sport_type,
                year=season, # Use 'year' for college sports if that's what fetch_matches_from_db expects
                team_name=team_name
            )
        else:
            return {"status": "error", "message": f"Unsupported sport type: {sport_type}"}

        if all_matches_df.empty:
            query_season_display = season if season else 'any season'
            return {"status": "no_matches", "message": f"No matches found for '{team_name or 'any team'}' in {sport_type} for season {query_season_display}."}

        # Convert DataFrame to a list of dicts for JSON serialization
        filtered_matches_list = all_matches_df.to_dict(orient="records")

        # Transform the flat DataFrame rows into the expected nested match JSON structure
        transformed_matches = []
        for match_row in filtered_matches_list:
            # Ensure all expected keys and nested structures are present
            transformed_matches.append({
                "fixture": {
                    "id": match_row.get("match_id"), 
                    "date": match_row.get("match_date"), 
                    "status": match_row.get("status")
                },
                "league": {
                    "name": match_row.get("league_name"), 
                    "season": match_row.get("season") # Assuming a 'season' column in your DB
                },
                "teams": {
                    "home": {"id": match_row.get("home_team_id"), "name": match_row.get("home_team_name")},
                    "away": {"id": match_row.get("away_team_id"), "name": match_row.get("away_team_name")}
                },
                "goals": { # Assuming goals are stored directly as 'home_score' and 'away_score'
                    "home": match_row.get("home_score"), 
                    "away": match_row.get("away_score")
                },
                "score": { # Assuming score details are similar
                    "fulltime": {"home": match_row.get("home_score"), "away": match_row.get("away_score")}
                },
                "sport_type": sport_type # Add sport_type for deep analysis later
            })
        # --- END FIX ---


        return {
            "status": "ok",
            "message": f"Found {len(transformed_matches)} matches. Select one for deep analysis.",
            "filtered_matches": transformed_matches, # Return the TRANSFORMED list
            "original_query_params": parsed_query_obj.dict()
        }

    def run_deep_analysis(self, selected_match: dict):
        """
        Runs the deep analysis pipeline for a single selected match.
        """
        match_id = selected_match["fixture"]["id"]
        sport_type = selected_match["sport_type"] # Make sure sport_type is available in selected_match
                                                # (It should be from the transformation in run method)

        # 1. Prediction Agent
        # --- FIX START: Use the 'invoke' method of the PredictionAgentLC ---
        prediction_output = self.prediction_agent.invoke(selected_match) 
        # --- FIX END ---
        
        # You might not need match_stats and odds_data directly for the XGBoost prediction itself,
        # but they are passed to other agents. Ensure these fetch calls are still here if needed.
        # For this specific error, we focus on prediction_output.

        # Fetch detailed stats and odds (keep these if other agents need them)
        match_stats_df = fetch_match_stats(match_id)
        odds_df = fetch_odds(match_id)

        # 2. Verification Agent
        # Pass necessary data for verification (e.g., prediction output and odds)
        verification_output = self.verification_agent.invoke({
            "match": selected_match, 
            "prediction": prediction_output, # Pass the output from our prediction agent
            "odds_data": odds_df.to_dict(orient="records") if not odds_df.empty else None
        })
        
        # 3. Behavior Agent (DQN Simulation Placeholder)
        behavior_input = {
            "raw_value_edge": verification_output.get('raw_value_edge', 0.0), 
            "confidence": verification_output.get('confidence', 'Low'),
            "user_risk_tolerance": "Medium" 
        }
        behavior_action = self.behavior_agent.invoke(behavior_input)
        
        # 4. Recommendation Agent (Synthesis)
        recommendation_output = self.recommendation_agent.invoke({
            "match": selected_match,
            "prediction_output": prediction_output,
            "verification_output": verification_output,
            "behavior_output": behavior_action,
        })
        
        # 5. Ethics Agent (Final Gate)
        ethics_output = self.ethics_agent.invoke(recommendation_output.get('recommendation_text', ''))

        return {
            "status": "ok",
            "message": "Deep analysis complete.",
            "prediction": prediction_output,
            "verification": verification_output,
            "action": behavior_action,
            "recommendation": recommendation_output,
            "ethics": ethics_output,
            "detailed_stats": match_stats_df.to_dict(orient="records") if not match_stats_df.empty else [],
            "detailed_odds": odds_df.to_dict(orient="records") if not odds_df.empty else []
        }