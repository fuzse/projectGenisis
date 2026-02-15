from ddgs import DDGS
from colorama import Fore, Style

def search_web(query):
    """
    Real-time web search using the 'ddgs' library.
    """
    print(Style.DIM + f"  → [Tool] Searching web for: '{query}'...")
    
    try:
        # The new library usage is cleaner
        results = DDGS().text(query, max_results=3)
        
        if not results:
            return "No results found."
            
        # Compress results for the AI
        summary = ""
        for r in results:
            summary += f"- {r['title']}: {r['body']}\n"
            
        return summary
    except Exception as e:
        print(Fore.RED + f"  [Tool Error] {e}")
        return "Search failed."
