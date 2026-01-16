from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from src.client import SpaceXClient

# --- Input Schemas ---
# We use Pydantic models to define the exact structure of inputs the LLM can provide.
# This prevents the "garbage in" problem where the LLM might hallucinate parameters.

class NextLaunchInput(BaseModel):
    """Input for getting the next launch. Requires no parameters."""
    pass

class LatestLaunchInput(BaseModel):
    """Input for getting the latest launch. Requires no parameters."""
    pass

class CompanyInfoInput(BaseModel):
    """Input for getting general company information about SpaceX."""
    pass

class RocketDetailsInput(BaseModel):
    rocket_id: str = Field(..., description="The unique identifier of the rocket (e.g., from a launch object).")

class LaunchpadDetailsInput(BaseModel):

    launchpad_id: str = Field(..., description="The unique identifier of the launchpad (e.g., from a launch object).")



class WikipediaInput(BaseModel):

    query: str = Field(..., description="The topic to search for on Wikipedia (e.g., 'SpaceX', 'Elon Musk').")



class LaunchQueryInput(BaseModel):

    year: Optional[int] = Field(None, description="Filter by year of launch (e.g., 2024).")

    success: Optional[bool] = Field(None, description="Filter by mission success status.")

    limit: Optional[int] = Field(5, description="Number of results to return (default 5).")



# --- Tool Implementations ---
# These functions act as the "Glue Code" between the clean Pydantic inputs
# and the actual API Client.

async def get_next_launch(client: SpaceXClient, _input: NextLaunchInput) -> Dict[str, Any]:

    """Fetches details about the next scheduled SpaceX launch."""

    return await client.get_next_launch()



async def get_latest_launch(client: SpaceXClient, _input: LatestLaunchInput) -> Dict[str, Any]:

    """Fetches details about the most recently completed SpaceX launch."""

    return await client.get_latest_launch()



async def get_company_info(client: SpaceXClient, _input: CompanyInfoInput) -> Dict[str, Any]:

    """Fetches general company information about SpaceX (CEO, location, valuation)."""

    return await client.get_company_info()



async def get_rocket_details(client: SpaceXClient, input_data: RocketDetailsInput) -> Dict[str, Any]:

    """Fetches technical details about a specific rocket using its ID."""

    return await client.get_rocket(input_data.rocket_id)



async def get_launchpad_details(client: SpaceXClient, input_data: LaunchpadDetailsInput) -> Dict[str, Any]:

    """Fetches details about a specific launchpad using its ID."""

    return await client.get_launchpad(input_data.launchpad_id)



async def get_wikipedia_summary(client: SpaceXClient, input_data: WikipediaInput) -> Dict[str, Any]:

    """Fetches a summary of a topic from Wikipedia."""

    return await client.get_wikipedia_summary(input_data.query)



async def query_launches(client: SpaceXClient, input_data: LaunchQueryInput) -> List[Dict[str, Any]]:

    """

    Searches for past launches based on filters like year and success.

    Returns a list of matching launches (limited to 5 by default).

    """

    query: Dict[str, Any] = {}

    

    if input_data.year is not None:

        # SpaceX API uses date_utc. We need to construct a range for the year.

        start_date = f"{input_data.year}-01-01T00:00:00.000Z"

        end_date = f"{input_data.year + 1}-01-01T00:00:00.000Z"

        query["date_utc"] = {"$gte": start_date, "$lt": end_date}

    

    if input_data.success is not None:

        query["success"] = input_data.success



    options = {

        "limit": input_data.limit,

        "sort": {"date_utc": "desc"}, # specificy sort order

        "select": ["name", "date_utc", "success", "rocket", "details", "failures"] # specific fields to save tokens

    }



    result = await client.query_launches(query, options)

    return result.get("docs", [])



# --- Tool Definitions for OpenAI ---
# This helper maps the Pydantic models to OpenAI's tool schema format.
# This allows us to keep the definitions in one place (Pydantic) and auto-generate the JSON schema.


def get_tool_schemas():

    return [

        {

            "type": "function",

            "function": {

                "name": "get_next_launch",

                "description": "Get information about the upcoming SpaceX launch.",

                "parameters": NextLaunchInput.model_json_schema(),

            }

        },

        {

            "type": "function",

            "function": {

                "name": "get_latest_launch",

                "description": "Get information about the last completed SpaceX launch.",

                "parameters": LatestLaunchInput.model_json_schema(),

            }

        },

        {

            "type": "function",

            "function": {

                "name": "get_company_info",

                "description": "Get general company information about SpaceX (CEO, valuation, etc).",

                "parameters": CompanyInfoInput.model_json_schema(),

            }

        },

        {

            "type": "function",

            "function": {

                "name": "get_rocket_details",

                "description": "Get details about a rocket (height, mass, description) by its ID.",

                "parameters": RocketDetailsInput.model_json_schema(),

            }

        },

        {

            "type": "function",

            "function": {

                "name": "get_launchpad_details",

                "description": "Get details about a specific SpaceX launchpad.",

                "parameters": LaunchpadDetailsInput.model_json_schema(),

            }

        },

        {

            "type": "function",

            "function": {

                "name": "get_wikipedia_summary",

                "description": "Get a summary of a topic from Wikipedia. Use this as a FALLBACK if SpaceX API fails or for general knowledge.",

                "parameters": WikipediaInput.model_json_schema(),

            }

        },

        {

            "type": "function",

            "function": {

                "name": "query_launches",

                "description": "Search for past launches by year or success status.",

                "parameters": LaunchQueryInput.model_json_schema(),

            }

        }

    ]