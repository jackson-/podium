# SpaceX Agent 🚀

A robust, CLI-based AI agent that answers questions about SpaceX using the SpaceX API. Designed to demonstrate ReAct patterns, including chat loops, tool usage, memory management, and fault tolerance.

## 🌟 Key Features

*   **Agentic Workflow:** Implements the **Reasoning -> Plan -> Act -> Observe** loop. The agent plans its approach, calls tools, and grounds its answers in real data.
*   **Multi-Turn Context:** Remembers conversation history to answer follow-up questions (e.g., "Who makes it?" after asking about Falcon 9).
*   **Robust Memory Management:** Features a **Sliding Window** context manager that intelligently prunes old messages while preserving the System Prompt and critical Tool/Observation chains to prevent API errors.
*   **Fault Tolerance:**
    *   **Retries:** Uses exponential backoff (`tenacity`) for handling API network blips.
    *   **Fallbacks:** Automatically queries Wikipedia if the SpaceX API lacks data or fails, ensuring the user almost always gets an answer.
*   **Temporal Awareness:** Dynamically injects the current date into the System Prompt, allowing the agent to understand "next launch" vs "last launch" relative to *now*.
*   **Type Safety:** Uses **Pydantic** models to strictly validate LLM tool inputs before execution.

## 🏗 System Architecture

### 1. The Agent (`src/agent.py`)
I chose to build the agent loop using the **raw OpenAI Client** rather than a framework like LangChain.
*   **Why?** Frameworks often hide complexity that is critical to control in production (e.g., how memory is pruned, how errors are handled). Building the loop manually demonstrates a deep understanding of the `messages` array, `tool_calls`, and the ReAct pattern. In a production setting, I would use a framework like LangChain or LangGraph.
*   **Memory:** The `_prune_memory` method ensures the context window never overflows, but uses heuristic logic to ensure we never slice the history in the middle of a `tool_call` -> `tool_result` chain, which would cause API errors.

### 2. The Client (`src/client.py`)
An asynchronous HTTP client built on `httpx`.
*   **Data Cleaning:** The SpaceX API returns massive JSON objects. I implemented a recursive `_clean_data` method to strip `null` values and unnecessary IDs. This reduces token usage by ~30-50% per call, improving latency and cost.
*   **Resilience:** Decorated with `@retry` to handle transient network failures (5xx, timeouts) automatically.

### 3. The Tools (`src/tools.py`)
Tools are the bridge between the LLM and the API.
*   **Explicit Schemas:** Every tool input is defined as a Pydantic model. This forces the LLM to adhere to a strict schema.
*   **Ambiguity Handling:** The System Prompt explicitly instructs the model to ask clarifying questions if a user's request is vague ("Tell me about the launch"), rather than guessing.

### 4. 🔮 Future Improvements

If this were going to production, I would add:
1.  **Vector Database (RAG):** Instead of relying on a live Wikipedia search, I would index the SpaceX documentation and use RAG for more technical questions.
2.  **Redis Caching:** Cache API responses (e.g., "latest launch" changes rarely) to respect rate limits and improve speed.
3.  **Structured Output:** Use `response_format={"type": "json_object"}` for the final answer to allow a frontend to render rich UI elements (like a Launch Card).
4.  **Telemetry:** Add OpenTelemetry tracing to visualize the full chain of thought and tool execution latency.

## 🛠 Setup & Usage

1.  **Install Dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Configure Environment:**
    Copy `.env.example` to `.env` and add your OpenAI API Key:
    ```bash
    cp .env.example .env
    # Edit .env with your OpenAI API key
    ```

3.  **Run the Agent:**
    ```bash
    python main.py
    ```

