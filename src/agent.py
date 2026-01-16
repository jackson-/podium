import os
import json
from datetime import datetime
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
    get_wikipedia_summary,
    query_launches,
    NextLaunchInput,
    LatestLaunchInput,
    RocketDetailsInput,
    CompanyInfoInput,
    LaunchpadDetailsInput,
    WikipediaInput,
    LaunchQueryInput
)

load_dotenv()

SYSTEM_PROMPT = """You are a specialized SpaceX Assistant.
Your goal is to answer questions about SpaceX using the provided tools.
Current Date: {date}

### RULES
1. **PLAN**: Before calling any tool, briefly state your plan (e.g., "I will fetch the latest launch to find the rocket ID, then query the rocket details.").
2. **CLARIFY**: If the user's query is vague (e.g., "tell me about the launch"), ask for clarification (e.g., "Which launch? The latest one, or a specific mission?").
3. **GROUNDING**: Only answer based on Tool Output. Do not hallucinate information or rely on internal knowledge which may be outdated.
4. **FALLBACK**: If a tool returns an error or data is missing, try to find the information using `get_wikipedia_summary`. If that also fails, explain the situation to the user.
5. **ANSWERING BOUNDARIES**: You can only answer questions about SpaceX. If the user asks about a different topic, redirect them to SpaceX.

### TOOLS
- Use `get_next_launch` for upcoming missions.
- Use `get_latest_launch` for the most recently completed mission.
- Use `get_company_info` for general SpaceX facts (CEO, headquarters, valuation).
- Use `query_launches` for filtering (e.g., "launches in 2024", "successful launches").
- Use `get_rocket_details` ONLY when you have a rocket ID (usually from a launch object) and need its name/specs.
- Use `get_launchpad_details` when you have a launchpad ID and need its location or name.
- Use `get_wikipedia_summary` as a fallback or for broader context not in the API.

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

    """

    The Core Agent Logic.

    

    Responsibilities:

    1.  **State Management**: Maintains the conversation history (`messages`).

    2.  **Memory Pruning**: Implements a sliding window to keep context manageable without breaking tool chains.

    3.  **Prompt Engineering**: Injects the current date and instructions into the System Prompt.

    4.  **ReAct Loop**: Handles the "Think -> Act -> Observe" cycle using OpenAI function calling.

    """

    def __init__(self):

        self.client = SpaceXClient()

        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        

        # Temporal Grounding:

        # Injects the current date into the system prompt. This allows the LLM to correctly

        # interpret relative time queries like "What launches are happening next week?".

        formatted_prompt = SYSTEM_PROMPT.format(date=datetime.now().strftime("%Y-%m-%d"))

        self.messages: List[Dict[str, Any]] = [

            {"role": "system", "content": formatted_prompt}

        ]

        

        # Tool Mapping: 

        # Maps the string name returned by OpenAI to the actual Python function and Pydantic model.

        self.tools_map = {

            "get_next_launch": get_next_launch,

            "get_latest_launch": get_latest_launch,

            "get_rocket_details": get_rocket_details,

            "get_company_info": get_company_info,

            "get_launchpad_details": get_launchpad_details,

            "get_wikipedia_summary": get_wikipedia_summary,

            "query_launches": query_launches

        }

        self.input_models = {

            "get_next_launch": NextLaunchInput,

            "get_latest_launch": LatestLaunchInput,

            "get_rocket_details": RocketDetailsInput,

            "get_company_info": CompanyInfoInput,

            "get_launchpad_details": LaunchpadDetailsInput,

            "get_wikipedia_summary": WikipediaInput,

            "query_launches": LaunchQueryInput

        }

        self.max_history = 20 # Keep last 20 messages to manage token usage vs context.



    def _prune_memory(self):

        """

        Maintains a sliding window of context while preserving tool call chains.

        

        Challenge:

        Simply slicing the list (`messages[-N:]`) can break the OpenAI API contract.

        A `tool` message (the observation) MUST immediately follow the `assistant` message 

        that requested it (`tool_calls`). If we slice between them, the API throws a 400 error.

        

        Solution:

        We iterate backwards to find the most recent 'user' message and cut the history 

        BEFORE that user message, effectively removing entire turn-pairs (User -> Assistant -> Tool -> Assistant).

        We ALWAYS preserve the System Prompt at index 0.

        """

        if len(self.messages) <= self.max_history:

            return



        # Start with the system prompt

        new_messages = [self.messages[0]]

        

        # Target length for the slice (excluding system prompt)

        target_len = self.max_history - 1

        

        # Initial slice of the most recent messages

        recent_messages = self.messages[-target_len:]

        

        # Find a safe cut point: the first 'user' message in the recent block.

        # This ensures we don't start in the middle of a tool chain.

        safe_start_index = 0

        for i, msg in enumerate(recent_messages):

            # msg can be a dict (user input) or an object (OpenAI response)

            role = msg.get('role') if isinstance(msg, dict) else msg.role

            

            if role == 'user':

                safe_start_index = i

                break

        

        # If we found a user message, trim everything before it in this slice

        final_slice = recent_messages[safe_start_index:]

        

        # Reconstruct: System Prompt + Safe Slice

        self.messages = new_messages + final_slice



    async def chat(self, user_input: str):

        """

        Main chat loop:

        1. Add user message to history.

        2. Call LLM (with tools enabled).

        3. Loop to handle tool calls:

           - If LLM wants to call a tool -> Execute it -> Add result to history -> Call LLM again.

           - If LLM returns text -> Print it and break the loop (wait for next user input).

        """

        self.messages.append({"role": "user", "content": user_input})

        self._prune_memory()



        while True:

            try:

                response = await self.openai.chat.completions.create(

                    model="gpt-4o",

                    messages=self.messages,

                    tools=get_tool_schemas(),

                    tool_choice="auto",

                    temperature=0.0 # Deterministic output preferred for tool usage

                )

            except Exception as e:

                cprint(f"Error calling OpenAI: {e}", "red")

                return



            message = response.choices[0].message

            self.messages.append(message)



            if message.content:

                # The model might plan (Reasoning) or answer here

                cprint(f"\nAssistant: {message.content}", "cyan")



            if not message.tool_calls:

                # No more tools needed, wait for user input

                break



            # Handle Tool Calls

            for tool_call in message.tool_calls:

                fn_name = tool_call.function.name

                args_json = tool_call.function.arguments

                tool_id = tool_call.id

                

                cprint(f"\n[Tool Call] {fn_name}({args_json})", "yellow")



                if fn_name in self.tools_map:

                    try:

                        # 1. Validation: Parse args using Pydantic

                        args_dict = json.loads(args_json)

                        model_class = self.input_models[fn_name]

                        validated_input = model_class(**args_dict)

                        

                        # 2. Execution: Run the actual tool

                        result = await self.tools_map[fn_name](self.client, validated_input)

                        

                        # 3. Observation: Add the result to history

                        cprint(f"[Observation] {str(result)[:100]}...", "green") # Truncate log for readability

                        self.messages.append({

                            "role": "tool",

                            "tool_call_id": tool_id,

                            "content": json.dumps(result)

                        })



                    except Exception as e:

                        # Error Handling: Pass the error back to the LLM so it can try again or explain it.

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

            

            # The while loop continues, sending the new tool outputs back to the LLM

            # to generate the next response or tool call.



    async def close(self):

        await self.client.close()