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
            max_tokens=750,
        )

        # 🔁 RISK-AWARE PROMPT – Adapts bet side based on player risk profile
        self.prompt = PromptTemplate.from_template("""You are a sports betting analyst. Provide a brief, tailored recommendation (120 words max).

MATCH: {home_team_name} vs {away_team_name} ({match_date}, {sport_type})
Status: {match_status} | {score_line_if_available}
Historic match (past result): {is_historical}

PROBABILITIES & EDGE:
Model: {predicted_winner_model} ({home_win_probability:.1%} H | {draw_probability:.1%} D | {away_win_probability:.1%} A)
Value edge: {raw_value_edge:.4f} ({value_edge_rating}) | Confidence: {confidence_level}
Safest bet: {safest_bet_side} ({safest_probability:.1%}) | Strategy: {recommendation_strategy}
YOUR BET: {recommended_bet_side}

PROFILE: {behavior_action} (risk tolerance: {behavior_risk_factor:.1f}/1.0)
Rule: LOW/MEDIUM→safer bet | HIGH/VALUE_BET→value bet

INSTRUCTIONS:
1. If is_historical={is_historical} (True), NO BET—explain models only.
2. If behavior_action=EXPLANATION_ONLY, describe numbers only.
3. Recommend {recommended_bet_side} using {recommendation_strategy} strategy.
4. If edge<0.01 AND confidence=Low, suggest PASS alternative.
5. End: "Only bet what you can afford to lose."

OUTPUT: Single paragraph with (1) match setup, (2) bet recommendation + strategy, (3) risk consideration, (4) responsible reminder.""")


        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        match_details = inputs.get("match", {})
        prediction_output = inputs.get("prediction_output", {})
        verification_output = inputs.get("verification_output", {})
        behavior_output = inputs.get("behavior_output", {})

        # --- Match status / score handling (same as before) ---
        fixture_status_info = match_details.get("fixture", {})
        match_status = fixture_status_info.get("status", "N/A")

        home_score_info = match_details.get("goals", {})
        home_score = home_score_info.get("home")
        away_score = home_score_info.get("away")

        score_line = ""
        if (
            match_status
            and match_status.lower() in ['finished', 'ft', 'full-time', 'match finished', 'completed']
            and home_score is not None and away_score is not None
        ):
            home_team_name_score = match_details.get('teams', {}).get('home', {}).get('name', 'Home Team')
            away_team_name_score = match_details.get('teams', {}).get('away', {}).get('name', 'Away Team')
            score_line = f"Final Score: {home_team_name_score} {home_score} - {away_score} {away_team_name_score}"
        elif match_status and match_status.lower() not in ['not started', 'tba', 'scheduled']:
            score_line = f"Match Status: {match_status.replace('_', ' ').title()}"
        else:
            score_line = "Match not yet started."

        # --- Historical vs future match flag (NEW) ---
        status_lower = (match_status or "").lower()
        is_historical = status_lower in [
            "finished",
            "ft",
            "full-time",
            "match finished",
            "completed",
            "postponed",   # optional; remove if you want postponed to still be "future"
        ]

        # --- Behavior action + risk factor extraction (NEW) ---
        behavior_action = "neutral_analysis"
        behavior_risk_factor = 0.5

        if isinstance(behavior_output, dict):
            # Prefer explicit "action" field, fall back to "bucket" if you use that naming
            behavior_action = behavior_output.get("action") or behavior_output.get("bucket") or "neutral_analysis"
            behavior_risk_factor = behavior_output.get("risk_factor", 0.5)
        elif isinstance(behavior_output, str):
            behavior_action = behavior_output

        # --- Handle "--" values for missing odds ---
        raw_edge_val = verification_output.get("raw_value_edge", 0.0)
        raw_edge_numeric = 0.0 if (isinstance(raw_edge_val, str) and raw_edge_val == "--") else float(raw_edge_val or 0.0)

        # --- SMART BET SIDE LOGIC: Risk-aware betting ---
        # 1. Identify SAFEST bet side (highest probability outcome)
        home_prob = prediction_output.get("home_win_probability", 0.0)
        draw_prob = prediction_output.get("draw_probability", 0.0)
        away_prob = prediction_output.get("away_win_probability", 0.0)

        home_team_name = match_details.get("teams", {}).get("home", {}).get("name", "Home")
        away_team_name = match_details.get("teams", {}).get("away", {}).get("name", "Away")
        sport_type = match_details.get("sport_type", "football")

        # Find safest outcome (highest probability)
        # NOTE: DRAW/TIE outcome is considered for soccer/football sports
        outcomes_with_prob = [
            (f"{home_team_name}_win", home_prob),
            (f"{away_team_name}_win", away_prob),
        ]

        # Check if sport supports DRAW/TIE (soccer/football)
        sport_lower = sport_type.lower()
        has_draw = "football" in sport_lower or "soccer" in sport_lower or sport_type in ["football", "soccer"]

        if has_draw and draw_prob > 0:
            outcomes_with_prob.append(("Draw", draw_prob))

        safest_bet_side = max(outcomes_with_prob, key=lambda x: x[1])[0]
        safest_prob = max(outcomes_with_prob, key=lambda x: x[1])[1]

        # 2. Get VALUE bet side (already from verification agent)
        value_bet_side = verification_output.get("recommended_bet_side", "None")

        # 3. Decide which to recommend based on risk profile
        # LOW/MEDIUM risk (0.0-0.67) → SAFEST
        # HIGH risk (0.67-1.0) or VALUE_BET action → VALUE
        risk_factor = behavior_risk_factor

        if behavior_action in ["SAFE_PICK", "EXPLANATION_ONLY"]:
            recommended_bet_side_final = safest_bet_side
            recommendation_strategy = "SAFE"
        elif behavior_action == "HIGH_RISK" or risk_factor > 0.67:
            recommended_bet_side_final = value_bet_side
            recommendation_strategy = "VALUE"
        elif behavior_action == "VALUE_BET" and risk_factor > 0.5:
            # VALUE_BET with moderate-high risk → go with value
            recommended_bet_side_final = value_bet_side
            recommendation_strategy = "VALUE"
        else:
            # MEDIUM risk (VALUE_BET with moderate risk) → SAFEST
            recommended_bet_side_final = safest_bet_side
            recommendation_strategy = "SAFE"

        # --- Prompt inputs ---
        prompt_inputs = {
            "home_team_name": home_team_name,
            "away_team_name": away_team_name,
            "match_date": match_details.get("fixture", {}).get("date", "N/A")[:10],
            "sport_type": sport_type,
            "match_status": match_status,
            "score_line_if_available": score_line,
            "is_historical": is_historical,

            "predicted_winner_model": prediction_output.get("predicted_winner_model", "N/A"),
            "home_win_probability": home_prob,
            "draw_probability": draw_prob,
            "away_win_probability": away_prob,

            "raw_value_edge": raw_edge_numeric,
            "value_edge_rating": verification_output.get("value_edge", "None"),
            "recommended_bet_side": recommended_bet_side_final,
            "safest_bet_side": safest_bet_side,
            "safest_probability": safest_prob,
            "recommendation_strategy": recommendation_strategy,
            "confidence_level": verification_output.get("confidence", "Low"),

            "behavior_action": behavior_action,
            "behavior_risk_factor": behavior_risk_factor,
        }

        recommendation_text = self.chain.invoke(prompt_inputs)

        # Return both the LLM text AND the risk-aware bet selection logic for display
        return {
            "recommendation_text": recommendation_text,
            "recommended_bet_side": recommended_bet_side_final,  # Risk-aware bet
            "recommendation_strategy": recommendation_strategy,   # SAFE or VALUE
            "safest_bet_side": safest_bet_side,                  # For reference
            "safest_probability": safest_prob,                   # Confidence in safest
        }
