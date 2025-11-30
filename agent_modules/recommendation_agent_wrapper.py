# /agent_modules/recommendation_agent_wrapper.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import OpenAI
import os
from typing import Dict, Any

class RecommendationAgentLC:
    def __init__(self):
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.7,
            max_tokens=750 # This is what you had, keep it or increase as discussed
        )

        self.prompt = PromptTemplate.from_template("""
            You are an AI sports betting analyst. Your goal is to provide a concise, clear, and ethical betting recommendation based on the provided analysis.

            --- Match Details ---
            Home Team: {home_team_name}
            Away Team: {away_team_name}
            Match Date: {match_date}
            Sport Type: {sport_type}
            Match Status: {match_status}
            {score_line_if_available}

            --- Prediction Model Output ---
            Predicted Winner (Model's Highest Probability): {predicted_winner_model}
            Home Win Probability: {home_win_probability:.2%}
            Draw Probability: {draw_probability:.2%}
            Away Win Probability: {away_win_probability:.2%}

            --- Value Verification Output ---
            Raw Value Edge: {raw_value_edge:.4f}
            Value Edge Rating: {value_edge_rating}
            Recommended Bet Side: {recommended_bet_side}
            Confidence Level: {confidence_level}

            --- Behavior Action ---
            Agent's Behavior Action: {behavior_action}

            --- Task ---
            Synthesize this information into a clear recommendation.
            1. State the teams involved, match status, and the predicted winner.
            2. If the match is finished, mention the final score.
            3. Explain the value edge, if any, and its rating.
            4. State the recommended bet side and confidence.
            5. Provide a final recommendation.
            6. Ensure the recommendation is ethical and responsible. If no value edge is found, recommend against betting.

            Final Recommendation (max 250 words):
        """)

        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        match_details = inputs.get("match", {})
        prediction_output = inputs.get("prediction_output", {})
        verification_output = inputs.get("verification_output", {})
        behavior_output = inputs.get("behavior_output", {})

        # Extract score information if available (for finished matches)
        # Using .get for nested dictionaries for safer access
        fixture_status_info = match_details.get("fixture", {})
        match_status = fixture_status_info.get("status", "N/A")
        
        home_score_info = match_details.get("goals", {})
        home_score = home_score_info.get("home")
        away_score = home_score_info.get("away")

        score_line = ""
        # Improved check for match status and scores
        if match_status and match_status.lower() in ['finished', 'ft', 'full-time', 'match finished', 'completed'] and \
           home_score is not None and away_score is not None:
            home_team_name_score = match_details.get('teams', {}).get('home', {}).get('name', 'Home Team')
            away_team_name_score = match_details.get('teams', {}).get('away', {}).get('name', 'Away Team')
            score_line = f"Final Score: {home_team_name_score} {home_score} - {away_score} {away_team_name_score}"
        elif match_status and match_status.lower() not in ['not started', 'tba', 'scheduled']:
            # For matches that are ongoing or postponed but not finished
            score_line = f"Match Status: {match_status.replace('_', ' ').title()}"
        else:
            score_line = "Match not yet started." # Or leave empty if preferred for future matches


        # Safely extract values, providing defaults
        prompt_inputs = {
            "home_team_name": match_details.get("teams", {}).get("home", {}).get("name", "N/A"),
            "away_team_name": match_details.get("teams", {}).get("away", {}).get("name", "N/A"),
            "match_date": match_details.get("fixture", {}).get("date", "N/A")[:10],
            "sport_type": match_details.get("sport_type", "N/A"),
            "match_status": match_status,
            "score_line_if_available": score_line,

            "predicted_winner_model": prediction_output.get("predicted_winner_model", "N/A"),
            "home_win_probability": prediction_output.get("home_win_probability", 0.0),
            "draw_probability": prediction_output.get("draw_probability", 0.0),
            "away_win_probability": prediction_output.get("away_win_probability", 0.0),

            "raw_value_edge": verification_output.get("raw_value_edge", 0.0),
            "value_edge_rating": verification_output.get("value_edge", "None"),
            "recommended_bet_side": verification_output.get("recommended_bet_side", "None"),
            "confidence_level": verification_output.get("confidence", "Low"),

            "behavior_action": behavior_output.get("action", "neutral_analysis")
        }

        # If behavior_output is a string directly, use it.
        if isinstance(behavior_output, str):
            prompt_inputs["behavior_action"] = behavior_output

        recommendation_text = self.chain.invoke(prompt_inputs)

        return {"recommendation_text": recommendation_text}