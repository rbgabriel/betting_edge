# /streamlit_app.py
import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from utils import (
    fetch_matches_from_db, 
    init_data_agent, 
    get_db_connection, 
    get_unique_leagues,
    fetch_match_stats,
    fetch_odds
)
import os
from data_agent import DataAgent
from dotenv import load_dotenv
from typing import Optional

# Existing query agent import (still used inside the pipeline)
from query_agent import parse_user_query

# New unified multi agent flow
from pipelines.pipeline import BettingEdgePipeline

# NEW: Odds API agent
from odds_agent import OddsAgent

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Betting Edge",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state from environment variables
if "api_key_football" not in st.session_state:
    st.session_state.api_key_football = os.getenv("API_KEY_FOOTBALL", "KEY_NOT_FOUND")
if "api_key_cfb" not in st.session_state:
    st.session_state.api_key_cfb = os.getenv("API_KEY_CFB", "KEY_NOT_FOUND")
if "api_key_basketball" not in st.session_state:
    st.session_state.api_key_basketball = os.getenv("API_KEY_BASKETBALL", "KEY_NOT_FOUND")

if "data_agent" not in st.session_state:
    st.session_state.data_agent = None
if "sport_type" not in st.session_state:
    st.session_state.sport_type = "football"
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False
# NEW: cache OddsAgent
if "odds_agent" not in st.session_state:
    st.session_state.odds_agent = None















# NEW: helper to map our sport_type -> Odds API sport key
# In streamlit_app.py, replace the mapping content:

def map_sport_to_odds_api(sport_type: str) -> str:
    mapping = {
        # FIX: Use the full, modern key for better compatibility
        "football": "soccer_england_premier_league", 
        "college_football": "americanfootball_ncaaf",
        "basketball": "basketball_nba",
    }
    # Note: If you encounter further 422 errors, you may need to update the CFB/CBB keys as well.
    return mapping.get(sport_type, "soccer_england_premier_league")


# NEW: helper to lazily init OddsAgent
def get_odds_agent() -> Optional[OddsAgent]:
    if st.session_state.odds_agent is None:
        try:
            st.session_state.odds_agent = OddsAgent()
        except Exception as e:
            st.error(f"Failed to initialize Odds API: {e}")
            st.session_state.odds_agent = None
    return st.session_state.odds_agent


# NEW: transform Odds API JSON into a DataFrame
def build_odds_dataframe(odds_data):
    if not odds_data:
        return pd.DataFrame()

    rows = []
    for event in odds_data:
        try:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            league = event.get("sport_title", event.get("sport_key", ""))
            commence = event.get("commence_time", "")
            try:
                dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                date_str = commence

            bookmakers = event.get("bookmakers", [])
            if not bookmakers:
                continue

            # Use first bookmaker with h2h market
            home_odds = draw_odds = away_odds = None
            bookmaker_name = bookmakers[0].get("title") or bookmakers[0].get("key")

            for market in bookmakers[0].get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name", "")
                        price = outcome.get("price", None)
                        if name == home:
                            home_odds = price
                        elif name == away:
                            away_odds = price
                        elif name.lower() == "draw":
                            draw_odds = price
                    break

            rows.append(
                {
                    "Date/Time": date_str,
                    "League": league,
                    "Home Team": home,
                    "Away Team": away,
                    "Bookmaker": bookmaker_name,
                    "Home Odds": home_odds,
                    "Draw Odds": draw_odds,
                    "Away Odds": away_odds,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


# Main App Header
st.markdown('<div class="main-header">⚽ Betting Edge</div>', unsafe_allow_html=True)
st.markdown("*AI-Powered Sports Intelligence System*")
st.divider()

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")

    sport_type = st.radio(
        "Select Sport",
        options=["football", "college_football", "basketball"],
        format_func=lambda x: {
            "football": "⚽ Soccer (Football-Data.org)",
            "college_football": "🏈 College Football",
            "basketball": "🏀 College Basketball",
        }.get(x),
        index=["football", "college_football", "basketball"].index(
            st.session_state.sport_type
        ),
    )

    # API Key Display
    if sport_type == "football":
        current_key = st.session_state.api_key_football
        st.text_input(
            "API Key",
            value="****" + current_key[-8:]
            if current_key != "KEY_NOT_FOUND"
            else "Not Set",
            disabled=True,
        )
    elif sport_type == "college_football":
        current_key = st.session_state.api_key_cfb
        st.text_input(
            "API Key",
            value="****" + current_key[-8:]
            if current_key != "KEY_NOT_FOUND"
            else "Not Set",
            disabled=True,
        )
    elif sport_type == "basketball":
        current_key = st.session_state.api_key_basketball
        st.text_input(
            "API Key",
            value="****" + current_key[-8:]
            if current_key != "KEY_NOT_FOUND"
            else "Not Set",
            disabled=True,
        )

    if st.button("🔌 Initialize Agent"):
        with st.spinner(f"Initializing {sport_type.replace('_', ' ').title()} Agent..."):
            # Call the init_data_agent from utils.py
            initialized_agent = init_data_agent(sport_type) 
            
            if initialized_agent:
                # Store the successfully initialized agent in session state
                st.session_state.data_agent = initialized_agent
                st.session_state.sport_type = sport_type # Also update sport_type
                st.session_state.db_initialized = True
                st.success(f"✅ {sport_type.replace('_', ' ').title()} Agent Connected!")
                st.rerun() # Rerun to display the tabs
            else:
                st.session_state.data_agent = None # Ensure it's None if initialization failed
                st.session_state.db_initialized = False
                st.error("Failed to initialize agent. Check API key and logs.")

    st.divider()

    # Database status
    st.subheader("📊 Database Status")
    if os.path.exists("betting_edge.db"):
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM matches")
        match_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM match_stats")
        stats_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM odds")
        odds_count = cursor.fetchone()[0]

        conn.close()

        st.metric("Matches", match_count)
        st.metric("Statistics", stats_count)
        st.metric("Odds Entries", odds_count)
    else:
        st.info("Database not yet initialized")

    st.divider()

    # Manual data fetching
    if st.session_state.data_agent:
        with st.expander("🛠️ Manual Data Tools"):
            st.subheader("Fetch Data Manually")

            if st.session_state.sport_type == "football":
                league_options = {
                    "Premier League": 39,
                    "La Liga": 140,
                    "Serie A": 135,
                    "Bundesliga": 78,
                    "Ligue 1": 61,
                }
                selected_league = st.selectbox(
                    "Select League", options=list(league_options.keys())
                )
                season = st.number_input(
                    "Season", min_value=2020, max_value=2025, value=2023
                )

                if st.button("📥 Fetch Matches"):
                    with st.spinner("Fetching..."):
                        try:
                            league_id = league_options[selected_league]
                            matches = st.session_state.data_agent.fetch_matches(
                                league_id=league_id, season=season
                            )
                            if matches:
                                count = 0
                                for m in matches:
                                    st.session_state.data_agent.store_match(m)
                                    count += 1
                                st.success(f"Stored {count} matches!")
                            else:
                                st.warning("No matches found.")
                        except Exception as e:
                            st.error(f"Error: {e}")

            else:
                year = st.number_input(
                    "Year", min_value=2020, max_value=2025, value=2024
                )
                if st.button("📥 Fetch Games"):
                    with st.spinner("Fetching..."):
                        try:
                            matches = st.session_state.data_agent.fetch_matches(
                                year=year
                            )
                            if matches:
                                count = 0
                                for m in matches:
                                    st.session_state.data_agent.store_match(m)
                                    count += 1
                                st.success(f"Stored {count} games!")
                            else:
                                st.warning("No games found.")
                        except Exception as e:
                            st.error(f"Error: {e}")


# Main content
if not st.session_state.data_agent:
    st.info("👋 Welcome! Please select a sport and click 'Initialize Agent' in the sidebar to begin.")
else:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🤖 AI Assistant", "🏠 Dashboard", "⚽ Matches", "📊 Statistics", "💰 Odds"]
    )

    # AI Assistant tab using unified pipeline
    with tab1:
        st.header("🤖 AI Sports Assistant")
        st.markdown(
            "Use natural language to query the multi agent flow. "
            "The system will parse your request, fetch data, run prediction, verification, behavior selection, recommendation, and ethics checks."
        )

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.info("Try: 'Fetch Premier League matches for Liverpool this season'")
        with col_ex2:
            st.info("Try: 'Get 2024 college basketball games for Duke'")

        user_query_tab1 = st.text_input( # Changed variable name to avoid conflict with top-level user_query
            "Ask the Assistant:", placeholder="Type your request here...", key="user_query_tab1"
        )
        
        # Initialize session state for pipeline results if not present
        if "pipeline_results" not in st.session_state:
            st.session_state.pipeline_results = None
        if "deep_analysis_results" not in st.session_state: # Ensure this is also initialized
            st.session_state.deep_analysis_results = None

        if st.button("🚀 Run Initial Query") and user_query_tab1: # Changed button text for clarity
            pipeline = BettingEdgePipeline() # Re-initialize pipeline for each run (or use cached if robust)

            with st.spinner("Running initial query and data agent..."):
                # Run the first part of the pipeline (Query -> Data Fetch -> Filter)
                result = pipeline.run(user_query_tab1)
                st.session_state.pipeline_results = result
                st.session_state.deep_analysis_results = None # Clear previous deep analysis if new initial query
            st.experimental_rerun() # Rerun to display initial results

        # Display initial results and offer deep analysis
        if st.session_state.pipeline_results:
            initial_query_result = st.session_state.pipeline_results

            if initial_query_result.get("status") == "ok":
                st.success(initial_query_result.get("message", "Initial query successful."))

                filtered_matches = initial_query_result.get("filtered_matches", [])
                if filtered_matches:
                    st.subheader("Select a Match for Deep Analysis:")
                    # Create a readable list for the selectbox
                    match_options = {
                        f"{m['teams']['home']['name']} vs {m['teams']['away']['name']} on {m['fixture']['date'][:10]} (ID: {m['fixture']['id']})": m
                        for m in filtered_matches
                    }
                    selected_match_key = st.selectbox(
                        "Choose a match:",
                        options=list(match_options.keys()),
                        key="match_selector"
                    )

                    if selected_match_key:
                        selected_match_for_analysis = match_options[selected_match_key]
                        
                        if st.button("▶️ Run Deep Analysis for Selected Match", key="run_deep_analysis_button"): # Changed button text
                            pipeline = BettingEdgePipeline() # Re-initialize for deep analysis
                            with st.spinner("Running prediction, verification, behavior, recommendation, and ethics agents..."):
                                # Call the deep analysis method of the pipeline
                                deep_analysis_results = pipeline.run_deep_analysis(selected_match_for_analysis)
                                st.session_state.deep_analysis_results = deep_analysis_results # Store deep analysis results
                            st.experimental_rerun() # Rerun to display deep analysis results

                else: # No filtered matches from initial query
                    st.warning(initial_query_result.get("message", "Agent understood the query, but found no matches."))
            
            elif initial_query_result.get("status") == "query_error":
                st.error(initial_query_result.get("message", "Query agent could not parse that request."))
            elif initial_query_result.get("status") == "no_matches":
                st.warning(initial_query_result.get("message", "No matches found from data agent for that sport/season."))
            else:
                st.error(initial_query_result.get("message", "An unexpected error occurred during the initial query phase."))

        # Display deep analysis results if available AND successfully run
        if 'deep_analysis_results' in st.session_state and st.session_state.deep_analysis_results:
            deep_analysis_result = st.session_state.deep_analysis_results

            if deep_analysis_result.get("status") == "ok":
                st.success("Deep Analysis Complete!")
                st.markdown("---")

                col1, col2 = st.columns(2)
                
                # --- LEFT COLUMN: MODEL PREDICTION ---
                with col1:
                    st.subheader("Prediction Model Output")
                    
                    # Get model outputs (safely)
                    # Check the keys against what PredictionAgentLC.invoke() returns
                    pred = deep_analysis_result.get('prediction', {}) 
                    
                    st.metric(
                        label=f"Winner (Highest %)",
                        # Make sure 'predicted_winner_model' is the exact key
                        value=pred.get('predicted_winner_model', 'N/A'), 
                        delta="Model Prediction"
                    )
                    st.metric(
                        label="Home Win Probability",
                        # Make sure 'home_win_probability' is the exact key
                        value=f"{pred.get('home_win_probability', 0.0):.1%}", 
                    )
                    st.metric(
                        label="Draw Probability",
                        # Make sure 'draw_probability' is the exact key
                        value=f"{pred.get('draw_probability', 0.0):.1%}", 
                    )
                    st.metric(
                        label="Away Win Probability",
                        # Make sure 'away_win_probability' is the exact key
                        value=f"{pred.get('away_win_probability', 0.0):.1%}", 
                    )

                # --- RIGHT COLUMN: VALUE VERIFICATION ---
                with col2:
                    st.subheader("Value Verification")
                    # Get verification outputs (safely)
                    # Check the keys against what OddsVerificationAgentLC.invoke() returns
                    verify = deep_analysis_result.get('verification', {}) 
                    
                    # NEW, CORRECTED LINE IN streamlit_app.py
                    st.metric(label="Raw Value Edge", value=f"{verify.get('raw_value_edge', 0.0):.2%}", delta=f"Rating: {verify.get('raw_value_edge', 'N/A')}") # Changed 'value_edge_rating' to 'value_edge'
                    st.metric(
                        label="Recommended Bet Side",
                        # Make sure 'recommended_bet_side' is the exact key
                        value=verify.get('recommended_bet_side', 'None'), 
                    )
                    st.metric(
                        label="Confidence Level",
                        # Make sure 'confidence' is the exact key
                        value=verify.get('confidence', 'Low'), 
                    )

                    # Display Behavior Action and Ethics Check here too for consistency
                    st.subheader("Behavior Action & Ethics")
                    
                    # Ensure action_output is retrieved, defaulting to an empty dict if None
                    action_output = deep_analysis_result.get('action', {}) 
                    
                    # --- THE FIX: Check if the output is a string (a low-level failure) ---
                    if isinstance(action_output, str):
                        behavior_action_display = action_output # Use the raw string error if it's not a dict
                    else:
                        # Safely access the dictionary key
                        behavior_action_display = action_output.get('action', 'neutral_not_found') 
                        
                    st.markdown(f"**Behavior Action:** {behavior_action_display}")
                    
                    ethics_output = deep_analysis_result.get('ethics', {})
                    st.markdown(f"**Ethics Check:** {ethics_output.get('status', 'pending')}")


                # --- FINAL RECOMMENDATION (LLM Narrative) ---
                st.divider()
                st.subheader("📝 Final Recommendation (LLM Synthesis)")
                recommendation_output = deep_analysis_result.get('recommendation', {})
                st.info(recommendation_output.get("recommendation_text", "No recommendation text available."))

                with st.expander("Debugging & Raw Agent Output"):
                    st.subheader("Full Prediction Output")
                    st.json(deep_analysis_result.get("prediction", {}))
                    
                    st.subheader("Full Verification Output")
                    st.json(deep_analysis_result.get("verification", {}))

                    # --- ADD THIS NEW DEBUGGING INFO ---
                    st.subheader("Raw selected_match passed to agents")
                    st.json(deep_analysis_result.get("match", {})) # Add this to see the match data
                    # --- END NEW DEBUGGING INFO ---

                    st.subheader("Full Recommendation Output")
                    st.json(deep_analysis_result.get("recommendation", {}))
                    
                    st.subheader("Full Ethics Output")
                    st.json(deep_analysis_result.get("ethics", {}))

                    st.subheader("Raw Value Edges (All Outcomes)")
                    st.json(deep_analysis_result.get("verification", {}).get("all_value_edges", {}))

            else: # Deep analysis returned an error status
                st.error(deep_analysis_result.get("message", "Deep analysis failed for an unknown reason."))
        
        # This part handles initial query errors and "no matches" from the initial pipeline.run()
        # It's crucial to have a separate check for initial_query_result to avoid trying to access
        # deep_analysis_result when only initial results are present or failed.
        if st.session_state.pipeline_results and st.session_state.pipeline_results.get("status") != "ok":
            if st.session_state.pipeline_results.get("status") == "query_error":
                st.error(st.session_state.pipeline_results.get("message", "Query agent could not parse that request."))
            elif st.session_state.pipeline_results.get("status") == "no_matches":
                st.warning(st.session_state.pipeline_results.get("message", "Agent understood the query, but found no matches involving that team or criteria."))
            else:
                st.error(st.session_state.pipeline_results.get("message", "An unexpected error occurred during the initial query phase."))


    # Dashboard tab
    with tab2:
        st.header(
            f"Dashboard Overview ({st.session_state.sport_type.replace('_', ' ').title()})"
        )

        if os.path.exists("betting_edge.db"):
            conn = get_db_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM matches WHERE match_date >= datetime('now', '-7 days') AND sport_type = ?",
                (st.session_state.sport_type,),
            )
            recent_matches = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(*) FROM matches WHERE match_date >= datetime('now') "
                "AND status NOT IN ('Match Finished', 'Match Cancelled', 'completed') "
                "AND sport_type = ?",
                (st.session_state.sport_type,),
            )
            upcoming_matches = cursor.fetchone()[0]

            cursor.execute(
                "SELECT COUNT(DISTINCT m.match_id) "
                "FROM odds o JOIN matches m ON o.match_id = m.match_id "
                "WHERE m.sport_type = ?",
                (st.session_state.sport_type,),
            )
            matches_with_odds = cursor.fetchone()[0]
            conn.close()

            col1, col2, col3 = st.columns(3)
            col1.metric("Recent Matches", recent_matches)
            col2.metric("Upcoming Matches", upcoming_matches)
            col3.metric("Matches with Odds", matches_with_odds)

            st.divider()
            st.subheader("📅 Latest Matches")

            leagues = get_unique_leagues(st.session_state.sport_type)
            col_f1, col_f2, col_f3, col_f4 = st.columns([2, 1, 1, 2])

            with col_f1:
                selected_league_filter = st.selectbox(
                    "Filter by League",
                    options=leagues,
                    key="dash_league_filter",
                )
            with col_f2:
                filter_past = st.checkbox("Past", value=True, key="dash_past")
            with col_f3:
                filter_future = st.checkbox("Future", value=True, key="dash_future")
            with col_f4:
                show_count = st.slider("Show", 10, 100, 20)

            matches_df = fetch_matches_from_db(
                sport_type=st.session_state.sport_type,
                league_name=selected_league_filter,
                include_past=filter_past,
                include_future=filter_future,
                limit=show_count,
            )

            if not matches_df.empty:
                matches_df["match_date"] = pd.to_datetime(
                    matches_df["match_date"]
                ).dt.strftime("%Y-%m-%d %H:%M")
                matches_df["Score"] = matches_df.apply(
                    lambda x: f"{int(x['home_score']) if pd.notna(x['home_score']) else 0} - "
                    f"{int(x['away_score']) if pd.notna(x['away_score']) else 0}",
                    axis=1,
                )
                st.dataframe(
                    matches_df[
                        [
                            "league_name",
                            "match_date",
                            "home_team_name",
                            "Score",
                            "away_team_name",
                            "status",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No matches in database. Use the AI Assistant or sidebar tools.")
        else:
            st.info("🔧 Database not found.")

    # Match details tab
    with tab3:
        st.header("Match Details")
        col1, col2 = st.columns(2)
        with col1:
            show_past = st.checkbox("Show Past Matches", value=True, key="tab3_past")
        with col2:
            show_future = st.checkbox(
                "Show Future Matches", value=True, key="tab3_future"
            )

        if os.path.exists("betting_edge.db"):
            matches_df = fetch_matches_from_db(
                sport_type=st.session_state.sport_type,
                league_name="All Leagues",
                include_past=show_past,
                include_future=show_future,
                limit=500,
            )

            if not matches_df.empty:
                match_options = matches_df.apply(
                    lambda x: f"{x['home_team_name']} vs {x['away_team_name']} ({x['match_date'][:10]})",
                    axis=1,
                ).tolist()

                selected_match_idx = st.selectbox(
                    "Select a match",
                    range(len(match_options)),
                    format_func=lambda x: match_options[x],
                )

                if selected_match_idx is not None:
                    selected_match = matches_df.iloc[selected_match_idx]
                    match_id = int(selected_match["match_id"])

                    c1, c2, c3 = st.columns([2, 1, 2])
                    c1.subheader(selected_match["home_team_name"])
                    c1.metric(
                        "Home",
                        int(selected_match["home_score"])
                        if pd.notna(selected_match["home_score"])
                        else "-",
                    )
                    c2.markdown(
                        "<h3 style='text-align: center;'>VS</h3>",
                        unsafe_allow_html=True,
                    )
                    c2.markdown(
                        f"<p style='text-align: center;'>{selected_match['status']}</p>",
                        unsafe_allow_html=True,
                    )
                    c3.subheader(selected_match["away_team_name"])
                    c3.metric(
                        "Away",
                        int(selected_match["away_score"])
                        if pd.notna(selected_match["away_score"])
                        else "-",
                    )

                    if st.session_state.sport_type == "football":
                        stats_df = fetch_match_stats(match_id)
                        if not stats_df.empty:
                            st.divider()
                            st.subheader("Statistics")
                            st.dataframe(stats_df, use_container_width=True)
            else:
                st.info("No matches available.")
        else:
            st.info("Database not found.")

    # Statistics tab
    with tab4:
        st.header("Team Statistics")
        if os.path.exists("betting_edge.db"):
            conn = get_db_connection()
            query = (
                "SELECT DISTINCT home_team_name as team_name, home_team_id as team_id "
                "FROM matches WHERE sport_type = ? "
                "UNION "
                "SELECT DISTINCT away_team_name, away_team_id FROM matches WHERE sport_type = ? "
                "ORDER BY team_name"
            )
            teams_df = pd.read_sql_query(
                query,
                conn,
                params=(st.session_state.sport_type, st.session_state.sport_type),
            )
            conn.close()

            if not teams_df.empty:
                selected_team = st.selectbox(
                    "Select Team", teams_df["team_name"].tolist(), key="stats_team_select"
                )
                if selected_team:
                    team_id = int(
                        teams_df[teams_df["team_name"] == selected_team]["team_id"].iloc[
                            0
                        ]
                    )
                    if st.session_state.data_agent:
                        recent = st.session_state.data_agent.get_recent_matches(team_id)
                        if recent:
                            st.subheader(f"Recent Form - {selected_team}")
                            wins = sum(
                                1
                                for m in recent
                                if (
                                    m["home_team_id"] == team_id
                                    and m["home_score"] > m["away_score"]
                                )
                                or (
                                    m["away_team_id"] == team_id
                                    and m["away_score"] > m["home_score"]
                                )
                            )
                            losses = len(recent) - wins
                            st.metric(
                                "Win Rate (Last 5)",
                                f"{(wins / len(recent) * 100):.0f}%",
                            )
                            st.dataframe(
                                pd.DataFrame(recent)[
                                    [
                                        "match_date",
                                        "home_team_name",
                                        "home_score",
                                        "away_score",
                                        "away_team_name",
                                    ]
                                ],
                                use_container_width=True,
                            )

    # Odds tab (UPDATED TO USE OddsAgent + mirror sport_type)
    with tab5:
        st.header("Betting Odds")

        odds_agent = get_odds_agent()
        if odds_agent is None:
            st.info("Unable to initialize Odds API. Check ODDS_API_KEY in your .env file.")
        else:
            current_sport_type = st.session_state.sport_type
            sport_key = map_sport_to_odds_api(current_sport_type)

            st.markdown(
                f"Showing upcoming odds for **{current_sport_type.replace('_', ' ').title()}** "
                f"(Odds API sport key: `{sport_key}`)"
            )

            regions = st.multiselect(
                "Regions",
                options=["us", "uk", "eu", "au"],
                default=["us", "eu"],
            )
            markets = st.multiselect(
                "Markets",
                options=["h2h", "ou", "spreads"],
                default=["h2h"],
            )

            if st.button("🔍 Fetch Latest Odds"):
                with st.spinner("Fetching odds from The Odds API..."):
                    odds_data = odds_agent.get_upcoming_odds(
                        sport=sport_key,
                        regions=",".join(regions) if regions else "us",
                        markets=",".join(markets) if markets else "h2h",
                    )

                if not odds_data:
                    st.warning("No odds data returned for this sport/region/market combination.")
                else:
                    odds_df = build_odds_dataframe(odds_data)
                    if odds_df.empty:
                        st.warning("Could not build a structured odds table from the API response.")
                    else:
                        st.subheader("Upcoming Odds (Sample)")
                        st.dataframe(odds_df, use_container_width=True, hide_index=True)

                    with st.expander("Raw Odds API Response"):
                        st.json(odds_data)
