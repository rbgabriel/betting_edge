# /agent_modules/query_agent.py (UPDATED)

import datetime
from typing import Optional, Literal
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# 1. Schema definition
class SportsDataQuery(BaseModel):
    sport_type: Literal["football", "college_football", "basketball"] = Field(
        ..., description="The sport to fetch. 'football' means Soccer."
    )
    team_name: Optional[str] = Field(
        None, description="Team to filter (e.g. 'Liverpool')"
    )
    competition_code: Optional[str] = Field(
        None, description="For Soccer only: PL, PD, SA, BL1, FL1, CL"
    )
    # --- FIX: Make season Optional ---
    season: Optional[int] = Field(
        None, description="YYYY season start. Defaults to current year if not specified."
    )
    # --- END FIX ---

# 2. LLM (unchanged)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Parser (unchanged)
parser = PydanticOutputParser(pydantic_object=SportsDataQuery)

# 4. Prompt (unchanged, but the "RULES" about "this season" will now be more important)
current_year = datetime.datetime.now().year

system_template = """
You are an expert sports data query translator.
Your job is to convert natural language questions into structured JSON parameters.

Current Year: {current_year}

RULES:
1. Soccer leagues -> codes: PL, PD, SA, BL1, FL1.
2. "College", "NCAA" -> college_football or basketball.
3. If user says "this season", set season to {current_year}. If "last season", set to {current_year_minus_1}.
4. Extract clean team names.
5. If a season is not explicitly mentioned (e.g., "Premier League matches for Liverpool"), DO NOT set the 'season' field. Leave it null.

{format_instructions}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{query}")
])

safe_prompt = prompt.partial(
    current_year=str(current_year),
    current_year_minus_1=str(current_year - 1),
    format_instructions=parser.get_format_instructions()
)

# 5. Query parser
def parse_user_query(user_query: str) -> SportsDataQuery:
    print(f"🤖 Query Agent: Analyzing '{user_query}'...")

    # old-LangChain compatible chain
    chain = safe_prompt | llm

    # run LLM
    response = chain.invoke({"query": user_query})

    # parse structured output
    structured = parser.parse(response.content)

    # --- FIX: Post-processing for season if LLM didn't provide it ---
    if structured.season is None:
        # This covers cases where the LLM might have missed "this season" or it wasn't explicit
        # For simplicity, default to current_year.
        # For football (soccer), current season might be year-1 if it spans two years (e.g., 2023-2024 is season 2023)
        # You might need to refine this based on your data agent's definition of 'season'.
        # Assuming current_year is good for now.
        structured.season = current_year 
        print(f"DEBUG: No season extracted by LLM, defaulting to {structured.season}")
    # --- END FIX ---

    print(f"✅ Query Agent: Parsed to {structured}")
    return structured