import asyncio
import sys
from termcolor import cprint
from src.agent import SpaceXAgent

async def main():
    cprint("🚀 SpaceX AI Agent Initialized (Type 'quit' or 'exit' to stop)", "magenta", attrs=["bold"])
    
    agent = SpaceXAgent()
    
    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ["quit", "exit"]:
                    cprint("Goodbye! 👋", "magenta")
                    break
                
                if not user_input:
                    continue
                    
                await agent.chat(user_input)
                
            except KeyboardInterrupt:
                cprint("\nInterrupted by user. Exiting...", "yellow")
                break
            except Exception as e:
                cprint(f"An unexpected error occurred: {e}", "red")
                break
    finally:
        await agent.close()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
