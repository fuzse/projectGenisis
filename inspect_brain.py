from memory.episodic import EpisodicMemory
from colorama import init, Fore, Style

init(autoreset=True)

def inspect():
    brain = EpisodicMemory()
    count = brain.count()
    print(Fore.CYAN + f"Total Memories: {count}")
    
    # 1. Peek at everything (limit to 10)
    print(Fore.YELLOW + "\n--- ALL MEMORIES ---")
    all_mems = brain.collection.get(limit=10)
    for i, doc in enumerate(all_mems['documents']):
        print(f"[{i}] {doc}")
        print(Style.DIM + f"    Meta: {all_mems['metadatas'][i]}")

    # 2. Test the specific retrieval that is failing
    query = "What is my favorite sports team?"
    print(Fore.YELLOW + f"\n--- SIMULATION: '{query}' ---")
    results = brain.recall(query, n_results=3)
    
    for mem in results:
        print(Fore.GREEN + f"RETRIVED: {mem['text']}")
        print(Style.DIM + f"Distance: {mem.get('dist', 'N/A')}")

if __name__ == "__main__":
    inspect()