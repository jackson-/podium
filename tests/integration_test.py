import asyncio
import sys
import os
from termcolor import cprint

# Add project root to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import SpaceXAgent

async def run_integration_tests():
    agent = SpaceXAgent()
    
    cprint("🚀 STARTING INTEGRATION TESTS 🚀", "cyan", attrs=["bold"])

    tests = [
        ("Context Retention", "What is the Falcon 9?", "Falcon 9"),
        ("Follow-up", "Who makes it?", "SpaceX"),
        ("Tool Usage (Wiki)", "Tell me about the history of the Starship SN8 test flight.", "SN8"),
        # Ambiguity: Look for "which" (Which launch?) or "specify" or "clarify"
        ("Ambiguity", "Tell me about the launch.", "which"), 
        # Error: Look for "no information" or "couldn't find" or "not found"
        ("Error Handling", "Tell me about the SpaceX 'Mars Imperial' rocket.", "couldn't find")
    ]

    for name, query, expected_keyword in tests:
        cprint(f"\n--- Test: {name} ---", "magenta")
        cprint(f"User: {query}", "white")
        
        await agent.chat(query)
        
        # VERIFICATION:
        # Check the last message in the agent's history to see if it matches expectations.
        last_msg = agent.messages[-1]
        last_response = last_msg.get("content") if isinstance(last_msg, dict) else last_msg.content
        
        if last_response and expected_keyword.lower() in last_response.lower():
            cprint(f"✅ PASS: Found keyword '{expected_keyword}'", "green")
        else:
            cprint(f"❌ FAIL: Expected '{expected_keyword}' but got:\n{str(last_response)[:100]}...", "red")
            # In a real CI/CD pipeline, we would raise an exception here.
            # raise Exception(f"Test {name} Failed")
    
    # Memory Pruning Test
    cprint("\n--- Test: Memory Pruning (Sliding Window) ---", "magenta")
    cprint("Injecting 25 dummy messages to force prune...", "yellow")
    for i in range(25):
        # Manually inject to avoid API costs/time, just testing the internal list logic
        agent.messages.append({"role": "user", "content": f"Dummy {i}"})
        agent.messages.append({"role": "assistant", "content": f"Response {i}"})
    
    # Trigger the prune
    agent._prune_memory()
    
    if len(agent.messages) <= agent.max_history:
        cprint(f"✅ Memory pruned successfully. Count: {len(agent.messages)}", "green")
    else:
        cprint(f"❌ Memory prune failed. Count: {len(agent.messages)}", "red")

    # Verify System Prompt is still index 0
    if agent.messages[0]['role'] == 'system':
        cprint("✅ System prompt preserved.", "green")
    else:
        cprint("❌ System prompt lost!", "red")

    await agent.close()
    cprint("\n✅ All Tests Finished", "green", attrs=["bold"])

if __name__ == "__main__":
    asyncio.run(run_integration_tests())
