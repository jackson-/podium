import os
import json
import logging
from typing import List, Dict, Any, Optional
from termcolor import cprint
from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.client import SpaceXClient
from src.tools import (
    get_tool_schemas,
    get_next_launch,
    get_latest_launch,
    get_rocket_details,
    get_company_info,
    get_launchpad_details,
    query_launches,
    NextLaunchInput,
    LatestLaunchInput,
    RocketDetailsInput,
    CompanyInfoInput,
    LaunchpadDetailsInput,
    LaunchQueryInput
)

load_dotenv()

SYSTEM_PROMPT = """You are a specialized SpaceX Assistant.
Your goal is to answer questions about SpaceX using the provided tools.

### RULES
1. **PLAN**: Before calling any tool, briefly state your plan (e.g., "I will fetch the latest launch to find the rocket ID, then query the rocket details.").
2. **CLARIFY**: If the user's query is vague (e.g., "tell me about the launch"), ask for clarification (e.g., "Which launch? The latest one, or a specific mission?").
3. **GROUNDING**: Only answer based on Tool Output. Do not hallucinate launch stats or rely on internal knowledge which may be outdated.
4. **FALLBACK**: If a tool returns an error, explain it to the user.

### TOOLS
- Use `get_next_launch` for upcoming missions.
- Use `get_latest_launch` for the most recently completed mission.
- Use `get_company_info` for general SpaceX facts (CEO, headquarters, valuation).
- Use `query_launches` for filtering (e.g., "launches in 2024", "successful launches").
- Use `get_rocket_details` ONLY when you have a rocket ID (usually from a launch object) and need its name/specs.
- Use `get_launchpad_details` when you have a launchpad ID and need its location or name.

### REASONING PROCESS
- If asked "Where was the last launch?", you must:
  1. Call `get_latest_launch` to get the `launchpad` ID.
  2. Call `get_launchpad_details` with that ID.
  3. Answer with the launchpad's name and location.
- If asked "Was the rocket in the last launch reusable?", you must:
  1. Call `get_latest_launch` to get the `rocket` ID.
  2. Call `get_rocket_details` with that ID.
  3. Answer based on the rocket's `description` or specs.
"""

class SpaceXAgent:
    def __init__(self):
        self.client = SpaceXClient()
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.tools_map = {
            "get_next_launch": get_next_launch,
            "get_latest_launch": get_latest_launch,
            "get_rocket_details": get_rocket_details,
            "get_company_info": get_company_info,
            "get_launchpad_details": get_launchpad_details,
            "query_launches": query_launches
        }
        self.input_models = {
            "get_next_launch": NextLaunchInput,
            "get_latest_launch": LatestLaunchInput,
            "get_rocket_details": RocketDetailsInput,
            "get_company_info": CompanyInfoInput,
            "get_launchpad_details": LaunchpadDetailsInput,
            "query_launches": LaunchQueryInput
        }

    async def chat(self, user_input: str):
        """
        Main chat loop:
        1. Add user message.
        2. Call LLM (with tools).
        3. Loop:
           - If tool calls: Execute -> Add result -> Call LLM again.
           - If content: Print and break.
        """
        self.messages.append({"role": "user", "content": user_input})

        while True:
            try:
                response = await self.openai.chat.completions.create(
                    model="gpt-4o",
                    messages=self.messages,
                    tools=get_tool_schemas(),
                    tool_choice="auto",
                    temperature=0.0
                )
            except Exception as e:
                cprint(f"Error calling OpenAI: {e}", "red")
                return

            message = response.choices[0].message
            self.messages.append(message)

            if message.content:
                # The model might plan or answer here
                cprint(f"\nAssistant: {message.content}", "cyan")

            if not message.tool_calls:
                break

            # Handle Tool Calls
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                args_json = tool_call.function.arguments
                tool_id = tool_call.id
                
                cprint(f"\n[Tool Call] {fn_name}({args_json})", "yellow")

                if fn_name in self.tools_map:
                    try:
                        # Parse args using Pydantic for validation
                        args_dict = json.loads(args_json)
                        model_class = self.input_models[fn_name]
                        validated_input = model_class(**args_dict)
                        
                        # Execute
                        result = await self.tools_map[fn_name](self.client, validated_input)
                        
                        # Add observation
                        cprint(f"[Observation] {str(result)[:100]}...", "green") # Truncate log
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps(result)
                        })

                    except Exception as e:
                        error_msg = f"Tool Execution Error: {str(e)}"
                        cprint(error_msg, "red")
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": json.dumps({"error": error_msg})
                        })
                else:
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": json.dumps({"error": f"Tool {fn_name} not found"})
                    })
            
            # Loop continues to send tool outputs back to LLM

    async def close(self):
        await self.client.close()