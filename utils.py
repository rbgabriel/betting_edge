# /utils.py

import streamlit as st
import sqlite3
import pandas as pd
import os
from data_agent import DataAgent # This is safe because DataAgent does NOT import from utils.py
from datetime import datetime
from typing import Optional

def init_data_agent(sport_type: str = "football") -> Optional[DataAgent]:
    """Initializes the DataAgent and handles errors."""
    try:
        agent = DataAgent(sport_type=sport_type)
        return agent
    except ValueError as e:
        # This error occurs if API key is missing
        st.error(f"Error initializing DataAgent for {sport_type}: {e}")
        return None
    except Exception as e:
        # General initialization error
        st.error(f"Failed to initialize Data Agent: {e}")
        return None

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect("betting_edge.db", check_same_thread=False)

def fetch_matches_from_db(
    sport_type: str,
    league_name: Optional[str] = None,
    year: Optional[int] = None, # This was for college sports, now `season` is more generic
    team_name: Optional[str] = None,
    season: Optional[int] = None, # <--- THIS MUST BE HERE AND BE OPTIONAL
    include_past: bool = True,
    include_future: bool = True,
    limit: int = 100,
) -> pd.DataFrame:
    """Fetch matches from database with filtering options."""
    conn = get_db_connection()

    params = [sport_type]
    conditions = ["sport_type = ?"]

    # Date filtering
    if not include_past:
        conditions.append("match_date >= datetime('now')")
    if not include_future:
        conditions.append("match_date < datetime('now')")

    if league_name and league_name != "All Leagues":
        conditions.append("league_name = ?")
        params.append(league_name)
    
    # --- IMPORTANT: Consolidate season/year filtering ---
    # For simplicity, let's use 'season' from now on for both.
    # If your DB column is strictly 'year' for college, we'd need a check.
    # Assuming 'season' is the primary filter column in your 'matches' table.
    if season: # Use the season from the query_agent
        conditions.append("season = ?")
        params.append(season)
    elif year: # Keep 'year' as a fallback if pipeline still sends it for college
        conditions.append("season = ?") # Assuming 'season' column
        params.append(year)
    # --- END CONSOLIDATION ---
    
    # Optional year filtering (for college sports)
    if year:
        conditions.append("season = ?")
        params.append(year)
    # --- ADD THIS BLOCK FOR TEAM NAME FILTERING ---
    if team_name:
        conditions.append("(home_team_name LIKE ? OR away_team_name LIKE ?)")
        params.append(f"%{team_name}%")
        params.append(f"%{team_name}%")
    # --- END ADDITION ---

    where_clause = f"WHERE {' AND '.join(conditions)}"

    query = f"""
        SELECT match_id, league_name, match_date,
               home_team_name, away_team_name,
               home_score, away_score, status
        FROM matches
        {where_clause}
        ORDER BY match_date DESC
        LIMIT ?
    """
    params.append(limit)

    try:
        df = pd.read_sql_query(query, conn, params=tuple(params))
        conn.close()
        return df
    except Exception as e:
        print(f"Error executing fetch_matches_from_db: {e}")
        conn.close()
        return pd.DataFrame()

# The other fetch functions (fetch_match_stats, fetch_odds) must also be moved here.

def fetch_match_stats(match_id: int):
    """Fetch statistics for a specific match. (Only uses existing simple columns)"""
    conn = get_db_connection()
    query = """
        SELECT team_name, shots_on_goal, total_shots, ball_possession
        FROM match_stats
        WHERE match_id = ?
    """
    try:
        df = pd.read_sql_query(query, conn, params=(match_id,))
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching stats: {e}")
        conn.close()
        return pd.DataFrame() # Return empty dataframe on error


def fetch_odds(match_id: int):
    """Fetch odds for a specific match (from local DB)."""
    conn = get_db_connection()
    query = """
        SELECT bookmaker, home_odds, draw_odds, away_odds
        FROM odds
        WHERE match_id = ?
    """
    try:
        df = pd.read_sql_query(query, conn, params=(match_id,))
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching odds: {e}")
        conn.close()
        return pd.DataFrame()

def get_unique_leagues(sport_type: str):
    """Get all unique leagues for the selected sport_type from the DB."""
    if not os.path.exists("betting_edge.db"):
        return ["All Leagues"]
    conn = get_db_connection()
    query = "SELECT DISTINCT league_name FROM matches WHERE sport_type = ? ORDER BY league_name"
    try:
        df = pd.read_sql_query(query, conn, params=(sport_type,))
        conn.close()
        return ["All Leagues"] + df["league_name"].tolist()
    except Exception as e:
        print(f"Error getting unique leagues: {e}")
        conn.close()
        return ["All Leagues"]