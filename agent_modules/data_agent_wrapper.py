# data_agent_lc.py
from langchain_core.runnables import Runnable
from data_agent import DataAgent
import os # Import os to get environment variables

class DataAgentLC(Runnable):
    """
    LangChain wrapper over your existing DataAgent.
    """

    def __init__(self):
        self.agent = None
        # Retrieve the Odds API key here to pass to DataAgent when it's initialized
        # This assumes ODDS_API_KEY is available in the environment when DataAgentLC is instantiated
        self.odds_api_key = os.getenv("ODDS_API_KEY", "KEY_NOT_FOUND")
        if self.odds_api_key == "KEY_NOT_FOUND":
            print("WARNING: ODDS_API_KEY not found in environment for DataAgentLC.")
            # Depending on your setup, you might want to raise an error here
            # or ensure the DataAgent handles a missing key gracefully.

    def invoke(self, params, **kwargs):
        sport = params["sport_type"]
        season = params["season"]
        comp = params.get("competition_code")
        team_name = params.get("team_name") # Also get team_name if query agent provides it

        # Instantiate DataAgent, passing the odds_api_key
        # This is the crucial change.
        try:
            self.agent = DataAgent(
                sport_type=sport,
                db_path="betting_edge.db",
                odds_api_key=self.odds_api_key # Pass the odds_api_key here
            )
        except ValueError as e:
            # Handle cases where DataAgent init fails due to missing keys
            print(f"ERROR: DataAgent initialization failed in DataAgentLC: {e}")
            return {"status": "error", "message": f"DataAgent initialization failed: {e}"}


        # The following logic is for *fetching* data using the DataAgent.
        # Your previous `invoke` implementation was attempting to directly call
        # `_fetch_football_data_org` or `fetch_matches`. The DataAgent's primary
        # role now is to coordinate.

        # The LC pipeline's `DataAgentLC` should return *matches* based on the query,
        # not just trigger a fetch. This means we should leverage `fetch_matches_from_db`
        # which is designed to query *existing* data in the DB.
        # If the intent is for the DataAgentLC to *fetch new data from external APIs*,
        # then the logic below would be more direct calls to `self.agent.fetch_matches`.
        # Assuming the pipeline's "initial query" primarily *retrieves* from DB
        # after potential fetching by the main Streamlit UI or a previous pipeline step:

        from utils import fetch_matches_from_db # Import here to avoid circular dependencies if utils imports DataAgentLC

        # Logic to fetch from the DB based on the parsed parameters
        # This aligns with the pipeline's `initial_query_agent` returning `filtered_matches`
        # that are likely stored in the database.
        try:
            matches_df = fetch_matches_from_db(
                sport_type=sport,
                league_name=comp, # Assuming competition_code maps to league_name
                season=season,
                team_name=team_name
            )

            if not matches_df.empty:
                # Convert DataFrame rows to list of dictionaries for consistent output
                matches_list = matches_df.to_dict(orient="records")
                return {"status": "ok", "message": "Matches found in DB.", "matches": matches_list}
            else:
                # If no matches found in DB, maybe we trigger an external fetch?
                # This depends on your pipeline design. For now, assume it's just DB lookup.
                return {"status": "no_matches", "message": "No matches found in DB for the specified criteria."}

        except Exception as e:
            print(f"Error fetching matches from DB in DataAgentLC: {e}")
            return {"status": "error", "message": f"Error retrieving matches: {e}"}