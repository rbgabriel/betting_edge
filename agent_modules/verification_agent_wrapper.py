# /agent_modules/verification_agent_wrapper.py

from langchain_core.runnables import Runnable
from data_agent import DataAgent
import math
from typing import Dict, Any

class VerificationAgentLC(Runnable):
    """
    Verification Agent: Calculates the mathematical Value Edge by comparing 
    Model Probability vs. Market Implied Probability (IP).
    """

    def __init__(self):
        # Initialize DataAgent here to access its methods (like fetch_odds)
        self.data_agent = DataAgent() 
        
    def _calculate_value(self, prediction_probs: Dict, market_odds: Dict, match_details: Dict) -> Dict:
        """
        Calculates the raw numerical value edge for Home, Draw, and Away,
        and assigns a qualitative rating based on the highest positive edge.
        
        CRITICAL CHANGE: Now also takes match_details to correctly map odds to teams.
        """
        
        # Get actual team names from match_details for clear mapping
        actual_home_team_name = match_details.get("teams", {}).get("home", {}).get("name")
        actual_away_team_name = match_details.get("teams", {}).get("away", {}).get("name")

        # 1. Convert Market Odds to Implied Probability (IP = 1 / Decimal Odds)
        # These are market odds FOR THE HOME TEAM, AWAY TEAM, and DRAW of THIS SPECIFIC MATCH.
        market_prob_home = 1 / market_odds.get('home_odds', 1000000) 
        market_prob_away = 1 / market_odds.get('away_odds', 1000000)
        market_prob_draw = 1 / market_odds.get('draw_odds', 1000000)
        
        # 2. Get Model Probabilities
        # These probabilities are for the 'home' side of the prediction (i.e., the first team in the pair)
        # and the 'away' side of the prediction (the second team in the pair).
        # These inherently correspond to the actual home and away teams of the 'match_details'.
        model_prob_home = prediction_probs.get('home_win_probability', 0.0)
        model_prob_away = prediction_probs.get('away_win_probability', 0.0)
        model_prob_draw = prediction_probs.get('draw_probability', 0.0)
        
        # 3. Calculate Value Edge (Value = Model Prob - Market IP) for all outcomes
        # Since prediction_probs and market_odds are both indexed by the *actual* home/away of the fixture,
        # the calculation directly aligns.
        value_home = model_prob_home - market_prob_home
        value_away = model_prob_away - market_prob_away
        value_draw = model_prob_draw - market_prob_draw

        all_value_edges = {
            f"{actual_home_team_name}_win": value_home, # Use actual team names for clarity
            "Draw": value_draw,
            f"{actual_away_team_name}_win": value_away
        }
        
        # 4. Determine the best bet_side based on the highest positive edge
        best_bet_side = "None"
        max_positive_edge = 0.0
        
        # Loop through outcomes to find the highest positive edge
        for outcome_label, edge_value in all_value_edges.items():
            if edge_value > max_positive_edge:
                max_positive_edge = edge_value
                best_bet_side = outcome_label # This will be like "Liverpool FC_win", "Draw", "West Ham United FC_win"
        
        # 5. Assign Qualitative Rating based on the *max_positive_edge*
        if max_positive_edge > 0.07:  # Over 7% edge 
            confidence = "High"
        elif max_positive_edge > 0.04:  # Over 4% edge
            confidence = "Medium"
        else:
            confidence = "Low" # No significant positive edge
            
        return {
            "value_edge_raw": float(round(max_positive_edge, 4)),
            "confidence": confidence, 
            "bet_side": best_bet_side, 
            "all_value_edges": {k: float(round(v, 4)) for k, v in all_value_edges.items()}
        }

    def invoke(self, inputs: Dict[str, Any], **kwargs) -> Dict:
        """
        Performs the verification step.
        Input: {'match': match_details, 'prediction': prediction_data}
        """
        match_details = inputs.get('match')
        prediction_data = inputs.get('prediction')

        # ... (Validation and Match ID Retrieval remains the same) ...
        if not match_details or match_details.get('status') == 'error':
             return {"value_edge": "Low", "confidence": "Low", "message": "Match details missing.", "raw_value_edge": 0.0, "recommended_bet_side": "None"}
             
        match_id = match_details['fixture']['id']

        # 2. Get Real-Time Odds
        # The DataAgent.fetch_odds method is the source of the 422 error, 
        # but we must call it here.
        market_odds = self.data_agent.fetch_odds(match_id)
        
        if not market_odds or market_odds.get('home_odds') is None:
            # Return a controlled error message that the Recommendation Agent can interpret
            return {"value_edge": "Low", "confidence": "Low", "message": "Market odds unavailable for comparison.", "raw_value_edge": 0.0, "recommended_bet_side": "None"}

        # 3. Calculate Value - PASS MATCH_DETAILS TO THE CALCULATION
        value_analysis = self._calculate_value(prediction_data, market_odds, match_details) # <--- CRITICAL CHANGE HERE
        
        # 4. Final Output (matching the required pipeline format)
        return {
            "value_edge": value_analysis['confidence'], 
            "confidence": value_analysis['confidence'], 
            "raw_value_edge": value_analysis['value_edge_raw'],
            "recommended_bet_side": value_analysis['bet_side'], # Use 'recommended_bet_side' for consistency
            "message": "Verification complete.",
            "all_value_edges": value_analysis['all_value_edges'] 
        }