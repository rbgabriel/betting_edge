# /query_agent.py — Enhanced NLP query parsing
#
# Improvements over v1:
#   • Pre-loads team/league vocab from DB at module init (no API round-trip)
#   • Injects vocab into LLM prompt so GPT-4o-mini can pick canonical names
#   • Post-processing fuzzy match: maps "Man City" → "Manchester City FC" etc.
#   • Fixture detection: "Arsenal vs Liverpool" → team_name + away_team_name
#   • Temporal resolution: "last season", "2023", "this year" → correct season int
#   • league name output = real DB league names, not opaque codes
#   • season left as None when user doesn't mention time (not defaulted to current year)

import re
import difflib
import sqlite3
import datetime
from typing import Optional, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = "betting_edge.db"
current_year = datetime.datetime.now().year

# ── Vocab: load canonical names from DB once at import time ──────────────────
def _load_vocab():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "SELECT DISTINCT home_team_name FROM matches WHERE home_team_name IS NOT NULL "
            "UNION "
            "SELECT DISTINCT away_team_name FROM matches WHERE away_team_name IS NOT NULL"
        )
        teams = sorted({r[0] for r in c.fetchall()})
        c.execute("SELECT DISTINCT league_name FROM matches WHERE league_name IS NOT NULL")
        leagues = sorted({r[0] for r in c.fetchall()})
        conn.close()
        return teams, leagues
    except Exception:
        return [], []

TEAM_VOCAB, LEAGUE_VOCAB = _load_vocab()

# ── Fuzzy matching ────────────────────────────────────────────────────────────
# Common aliases that difflib alone won't resolve
_TEAM_ALIASES = {
    "man city": "Manchester City FC",
    "man utd": "Manchester United FC",
    "man united": "Manchester United FC",
    "arsenal": "Arsenal FC",
    "liverpool": "Liverpool FC",
    "chelsea": "Chelsea FC",
    "spurs": "Tottenham Hotspur FC",
    "tottenham": "Tottenham Hotspur FC",
    "barca": "FC Barcelona",
    "barcelona": "FC Barcelona",
    "real madrid": "Real Madrid CF",
    "atletico": "Club Atlético de Madrid",
    "atletico madrid": "Club Atlético de Madrid",
    "psg": "Paris Saint-Germain FC",
    "paris sg": "Paris Saint-Germain FC",
    "paris saint germain": "Paris Saint-Germain FC",
    "juventus": "Juventus FC",
    "inter milan": "FC Internazionale Milano",
    "inter": "FC Internazionale Milano",
    "ac milan": "AC Milan",
    "milan": "AC Milan",
    "ajax": "AFC Ajax",
    "porto": "FC Porto",
    "benfica": "SL Benfica",
    "celtic": "Celtic FC",
    "rangers": "Rangers FC",
    "dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "leverkusen": "Bayer 04 Leverkusen",
    "bayern": "FC Bayern München",
    "munich": "FC Bayern München",
    "frankfurt": "Eintracht Frankfurt",
    "napoli": "SSC Napoli",
    "roma": "AS Roma",
    "lazio": "SS Lazio",
    "sevilla": "Sevilla FC",
    "villarreal": "Villarreal CF",
    "valencia": "Valencia CF",
    "athletic bilbao": "Athletic Club",
    "bilbao": "Athletic Club",
    "osasuna": "CA Osasuna",
    "alaves": "Deportivo Alavés",
    "girona": "Girona FC",
    "brighton": "Brighton & Hove Albion FC",
    "newcastle": "Newcastle United FC",
    "aston villa": "Aston Villa FC",
    "west ham": "West Ham United FC",
    "brentford": "Brentford FC",
    "fulham": "Fulham FC",
    "everton": "Everton FC",
    "wolves": "Wolverhampton Wanderers FC",
    "wolverhampton": "Wolverhampton Wanderers FC",
    "crystal palace": "Crystal Palace FC",
    "nottingham forest": "Nottingham Forest FC",
    "forest": "Nottingham Forest FC",
    "bournemouth": "AFC Bournemouth",
    "burnley": "Burnley FC",
    "leeds": "Leeds United FC",
    "club brugge": "Club Brugge KV",
    "brugge": "Club Brugge KV",
    "monaco": "AS Monaco FC",
    "galatasaray": "Galatasaray SK",
    "copenhagen": "FC København",
    "bodo glimt": "FK Bodø/Glimt",
}

_LEAGUE_ALIASES = {
    "pl": "Premier League",
    "epl": "Premier League",
    "premier league": "Premier League",
    "la liga": "Primera Division",
    "primera division": "Primera Division",
    "pd": "Primera Division",
    "sa": "Serie A",
    "serie a": "Serie A",
    "bundesliga": "Bundesliga",
    "bl1": "Bundesliga",
    "ligue 1": "Ligue 1",
    "fl1": "Ligue 1",
    "champions league": "UEFA Champions League",
    "ucl": "UEFA Champions League",
    "cl": "UEFA Champions League",
    "uefa champions league": "UEFA Champions League",
}


def fuzzy_match_team(raw: str, cutoff: float = 0.55) -> Optional[str]:
    """Map a raw team string to the best canonical DB team name."""
    if not raw:
        return raw
    key = raw.lower().strip()

    # 1. Alias lookup (instant, handles the most common cases)
    if key in _TEAM_ALIASES:
        return _TEAM_ALIASES[key]

    # 2. Exact substring match against DB vocab (e.g. "Bayern" ⊂ "FC Bayern München")
    for name in TEAM_VOCAB:
        if key in name.lower() or name.lower() in key:
            return name

    # 3. difflib fuzzy match as last resort
    if TEAM_VOCAB:
        vocab_lower = [t.lower() for t in TEAM_VOCAB]
        matches = difflib.get_close_matches(key, vocab_lower, n=1, cutoff=cutoff)
        if matches:
            return TEAM_VOCAB[vocab_lower.index(matches[0])]

    return raw  # return original if truly unrecognised


def fuzzy_match_league(raw: str) -> Optional[str]:
    """Map a raw league string to the best canonical DB league name."""
    if not raw:
        return raw
    key = raw.lower().strip()

    # 1. Alias lookup
    if key in _LEAGUE_ALIASES:
        return _LEAGUE_ALIASES[key]

    # 2. Substring match against DB vocab
    for name in LEAGUE_VOCAB:
        if key in name.lower() or name.lower() in key:
            return name

    # 3. difflib
    if LEAGUE_VOCAB:
        vocab_lower = [l.lower() for l in LEAGUE_VOCAB]
        matches = difflib.get_close_matches(key, vocab_lower, n=1, cutoff=0.6)
        if matches:
            return LEAGUE_VOCAB[vocab_lower.index(matches[0])]

    return raw


# ── Temporal expression resolver ─────────────────────────────────────────────
_YEAR_RE = re.compile(r"\b(20\d{2})\b")

def resolve_temporal(query: str) -> Optional[int]:
    """
    Returns a season year int if the query contains a temporal expression,
    or None if no season/time reference is present.
    Deliberately returns None for timeless queries like "Liverpool matches".
    """
    q = query.lower()
    if any(p in q for p in ["this season", "current season", "this year"]):
        return current_year
    if any(p in q for p in ["last season", "previous season", "last year"]):
        return current_year - 1
    # Explicit 4-digit year in query
    m = _YEAR_RE.search(query)
    if m:
        return int(m.group(1))
    # "recent" / "latest" → still no season filter (let DB return latest)
    return None


# ── Fixture pattern detector ──────────────────────────────────────────────────
_FIXTURE_RE = re.compile(
    r"^(?:.*?)\b(.+?)\s+(?:vs\.?|versus|v\.?|against|[-–])\s+(.+?)(?:\s+(?:in|on|at|this|last|next|for|\d{4}).*)?$",
    re.IGNORECASE,
)

def detect_fixture(query: str):
    """
    Returns (home_raw, away_raw) tuple if a 'X vs Y' pattern is found.
    Strips common boilerplate words from either side.
    Returns None if no fixture pattern detected.
    """
    m = _FIXTURE_RE.search(query.strip())
    if m:
        home_raw = m.group(1).strip()
        away_raw = m.group(2).strip()
        # Strip leading noise words
        for noise in ("fetch", "get", "show", "find", "match", "game", "fixture", "between"):
            home_raw = re.sub(rf"^{noise}\s+", "", home_raw, flags=re.IGNORECASE).strip()
        return home_raw, away_raw
    return None


# ── Schema ────────────────────────────────────────────────────────────────────
class SportsDataQuery(BaseModel):
    sport_type: Literal["football", "college_football", "basketball"] = Field(
        ..., description="Sport type. 'football' means Soccer/Association Football."
    )
    team_name: Optional[str] = Field(
        None,
        description="Primary team name exactly as it appears in the DB vocab, or as the user typed it."
    )
    away_team_name: Optional[str] = Field(
        None,
        description="Second team name for fixture queries (e.g. 'Arsenal vs Liverpool'). Leave null if only one team mentioned."
    )
    competition_code: Optional[str] = Field(
        None,
        description=(
            "Full league name as stored in the database. "
            "Use: 'Premier League', 'Primera Division', 'Serie A', 'Bundesliga', 'Ligue 1', "
            "'UEFA Champions League'. Leave null if not mentioned."
        )
    )
    season: Optional[int] = Field(
        None,
        description=(
            "Season start year as YYYY integer. "
            "Set ONLY when the user explicitly mentions a year or season (e.g. '2023', 'last season'). "
            "Leave null for timeless queries like 'Liverpool matches'."
        )
    )


# ── LLM + parser setup ────────────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
parser = PydanticOutputParser(pydantic_object=SportsDataQuery)

# Build a compact vocab string to inject into the prompt
_team_vocab_hint = (
    ", ".join(TEAM_VOCAB[:80]) if TEAM_VOCAB
    else "Liverpool FC, Arsenal FC, Manchester City FC, FC Barcelona, Real Madrid CF"
)
_league_vocab_hint = (
    ", ".join(LEAGUE_VOCAB) if LEAGUE_VOCAB
    else "Premier League, Primera Division, UEFA Champions League"
)

system_template = """You are an expert sports query parser.
Convert the user's natural-language query into structured JSON.

Current year: {current_year}

KNOWN TEAMS IN DATABASE (use these as reference):
{team_vocab}

KNOWN LEAGUES IN DATABASE (output EXACTLY one of these strings, or null):
{league_vocab}

RULES:
1. sport_type: "football" = soccer. "college_football" or "basketball" for NCAA.
2. competition_code: output the FULL league name from the list above (e.g. "Premier League", not "PL"). Null if not mentioned.
3. team_name / away_team_name: for "X vs Y" queries fill BOTH fields. For single-team queries fill only team_name.
4. season: set ONLY when the user explicitly mentions a year or time ("2023", "last season", "this year"). Leave NULL otherwise.
5. Clean team names: remove words like "matches", "games", "fixtures", "fetch", "get".
6. "this season" / "current season" → {current_year}. "last season" → {current_year_minus_1}.

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{query}")
])

safe_prompt = prompt.partial(
    current_year=str(current_year),
    current_year_minus_1=str(current_year - 1),
    team_vocab=_team_vocab_hint,
    league_vocab=_league_vocab_hint,
    format_instructions=parser.get_format_instructions(),
)


# ── Main entry point ──────────────────────────────────────────────────────────
def parse_user_query(user_query: str) -> SportsDataQuery:
    print(f"🔍 QueryAgent: '{user_query}'")

    # --- Pre-processing: fixture detection & temporal hint ---
    fixture = detect_fixture(user_query)
    season_hint = resolve_temporal(user_query)

    # --- LLM parse ---
    chain = safe_prompt | llm
    response = chain.invoke({"query": user_query})
    structured = parser.parse(response.content)

    # --- Post-processing: fuzzy match team names ---
    if structured.team_name:
        structured.team_name = fuzzy_match_team(structured.team_name)

    if structured.away_team_name:
        structured.away_team_name = fuzzy_match_team(structured.away_team_name)

    # --- Apply fixture detection if LLM missed the second team ---
    if fixture and not structured.away_team_name:
        home_raw, away_raw = fixture
        if not structured.team_name:
            structured.team_name = fuzzy_match_team(home_raw)
        structured.away_team_name = fuzzy_match_team(away_raw)

    # --- Fuzzy match league name ---
    if structured.competition_code:
        structured.competition_code = fuzzy_match_league(structured.competition_code)

    # --- Season: use temporal hint if LLM left it null, but don't force current_year ---
    if structured.season is None and season_hint is not None:
        structured.season = season_hint
    # If still None after all processing, leave it None (pipeline has fallbacks)

    print(f"✅ QueryAgent parsed → {structured}")
    return structured
