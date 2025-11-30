import requests
import os
from dotenv import load_dotenv
from typing import Optional # Import Optional for type hinting

load_dotenv()

class OddsAgent:
    """
    Fetches betting odds (future + live) using The Odds API.
    Docs: https://the-odds-api.com/
    """

    # Add api_key as a parameter with a default of None, making it optional
    # but allowing explicit passing from Streamlit.
    def __init__(self, api_key: Optional[str] = None):
        # If an api_key is passed directly, use it. Otherwise, try to load from .env.
        self.api_key = api_key if api_key else os.getenv("ODDS_API_KEY")

        if not self.api_key:
            # Raise ValueError if key is still not found after checking both sources
            raise ValueError("❌ Missing ODDS_API_KEY. Please provide it or set it in your .env file.")
        
        self.base_url = "https://api.the-odds-api.com/v4"

    def _get(self, endpoint, params):
        # Ensure the API key is always part of the parameters for the request
        params_with_key = params.copy() # Create a copy to avoid modifying the original params dict
        params_with_key["apiKey"] = self.api_key

        # --- NEW DEBUG PRINT ---
        full_url = f"{self.base_url}{endpoint}"
        print(f"DEBUG: OddsAgent attempting to call URL: {full_url} with params: {params_with_key}")
        # --- END NEW DEBUG PRINT ---

        try:
            response = requests.get(f"{self.base_url}{endpoint}", params=params_with_key)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            print(f"❌ Odds API HTTP Error: {http_err}")
            print(f"Response content: {response.text}") # Print response body for more info
            return []
        except requests.exceptions.ConnectionError as conn_err:
            print(f"❌ Odds API Connection Error: {conn_err}")
            return []
        except requests.exceptions.Timeout as timeout_err:
            print(f"❌ Odds API Timeout Error: {timeout_err}")
            return []
        except requests.exceptions.RequestException as req_err:
            print(f"❌ Odds API Request Error: {req_err}")
            return []
        except Exception as e:
            print(f"❌ An unexpected error occurred in Odds API _get: {e}")
            return []

    # ----------------------------
    # UPCOMING / FUTURE ODDS
    # ----------------------------
    # Changed default 'markets' from 'ou' to 'totals' for consistency.
    def get_upcoming_odds(self, sport="soccer_epl", regions="us,eu", markets="h2h,totals,spreads"):
        # --- NEW DEBUG PRINT ---
        print(f"DEBUG: OddsAgent.get_upcoming_odds called with sport='{sport}', regions='{regions}', markets='{markets}'")
        # --- END NEW DEBUG PRINT ---
        params = {
            "regions": regions,
            "markets": markets,
            "oddsFormat": "decimal",
            "dateFormat": "iso" # Ensure ISO format for easy parsing in Streamlit
        }
        endpoint = f"/sports/{sport}/odds"
        return self._get(endpoint, params)

    # ----------------------------
    # LIVE SCORES + LIVE ODDS
    # ----------------------------
    def get_live_odds(self, sport="soccer_epl"):
        """
        Odds API supports recent scores via the /scores endpoint, typically for
        events that have recently commenced or completed. Not true real-time in-play odds.
        """
        params = {
            "daysFrom": 3,
            "dateFormat": "iso"
        }
        endpoint = f"/sports/{sport}/scores" # Note: this is for scores, not odds directly
        return self._get(endpoint, params)

    # ----------------------------
    # LIST AVAILABLE SPORTS
    # ----------------------------
    def list_sports(self):
        endpoint = "/sports"
        params = {} # No specific params needed for listing sports
        return self._get(endpoint, params)