from memory.episodic import EpisodicMemory
from colorama import init, Fore, Style

init(autoreset=True)

def test_brain():
    print(Fore.YELLOW + "--- Initializing Memory Core ---")
    brain = EpisodicMemory()
    
    # 1. Check if it's empty
    initial_count = brain.count()
    print(f"Initial Memory Count: {initial_count}")
    
    # 2. Add a test memory if it's new
    if initial_count == 0:
        print(Fore.CYAN + "Injecting first memory...")
        brain.add_memory(
            text="The user's name is Architect. He is building a cognitive AI.",
            metadata={"type": "fact", "importance": "high"}
        )
        brain.add_memory(
            text="The user likes the Denver Broncos.",
            metadata={"type": "preference", "sentiment": "positive"}
        )
    
    # 3. Test Retrieval
    query = "What sports team does the user like?"
    print(Fore.YELLOW + f"\nQuerying: '{query}'")
    
    results = brain.recall(query)
    
    for mem in results:
        print(Fore.GREEN + f"Found: {mem['text']}")
        print(Style.DIM + f"Metadata: {mem['meta']}")

if __name__ == "__main__":
    test_brain()