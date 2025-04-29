import asyncio
from claude3_7 import ClaudeSonnetCodeAssistant
from rich.console import Console

async def main():
    # Instantiate without API key for diff testing (should not hit API)
    assistant = ClaudeSonnetCodeAssistant(api_key="dummy-api-key")
    # Example texts to diff
    original = """def hello():
    print("Hello, world!")

def foo():
    return 42
"""
    new = """def hello():
    print("Hello, Universe!")

def foo():
    return 42

def bar():
    print("New function")
"""
    # Produce colored diff
    panel = await assistant.create_colored_diff(
        original, new, "example.py", enhanced=True
    )
    # Render the Panel
    console = Console()
    console.print(panel)

if __name__ == "__main__":
    asyncio.run(main())