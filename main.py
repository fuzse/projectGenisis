import ollama
from colorama import Fore, Style, init
import config
from memory.episodic import EpisodicMemory
from memory.sleep import go_to_sleep
from core.wakeup import run_wakeup_routine
from core.perception import check_intent
from core.tools import search_web

# Initialize colors
init(autoreset=True)

def main():
    print(Fore.CYAN + f"Initializing {config.AI_NAME} on {config.MODEL_NAME}...")
    
    # 1. Load Memory
    brain = EpisodicMemory()
    print(Fore.GREEN + f"✔ Hippocampus Linked ({brain.count()} memories)")

    # --- WAKE UP ROUTINE ---
    greeting = run_wakeup_routine()
    print(Fore.GREEN + f"\n{config.AI_NAME}: {greeting}")
    
    # Initialize history with the greeting so the AI knows it spoke first
    chat_history = [
        {'role': 'assistant', 'content': greeting}
    ]

    print(Fore.YELLOW + "\nSystem Online. (Type 'exit' to quit)")
    
    while True:
        user_input = input(Fore.WHITE + f"\n{config.USER_NAME}: ")
        
        if user_input.lower() in ["exit", "quit"]:
            # Trigger the Sleep Cycle before shutting down
            go_to_sleep(chat_history)
            print(Fore.CYAN + "Shutting down...")
            break
            
        # --- 1. PERCEPTION LAYER (The Router) ---
        # "Does this input require a tool?"
        intent = check_intent(user_input)
        tool_result = ""
        
        if "SEARCH:" in intent:
            print(Style.DIM + f"  → Intent Detected: {intent}")
            # Extract the query
            query = intent.replace("SEARCH:", "").strip()
            # Run the tool
            tool_result = search_web(query)
            print(Fore.CYAN + f"  → Search Results Acquired.")
        
        # --- 2. MEMORY RECALL ---
        # "Have I heard about this before?"
        memories = brain.recall(user_input, n_results=2)
        
        # Format memories into a string
        memory_context = ""
        if memories:
            memory_context = "LONG TERM MEMORY:\n"
            for m in memories:
                memory_context += f"- {m['text']}\n"
        
        # --- 3. CONSTRUCT PROMPT ---
        # We combine Identity + Memory + Web Results + User Input
        system_prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are {config.AI_NAME}. You are chatting with {config.USER_NAME}.
        
        CONTEXT FROM MEMORY:
        {memory_context if memory_context else "No relevant memories found."}
        
        WEB SEARCH RESULTS:
        {tool_result if tool_result else "No search performed."}
        
        INSTRUCTIONS:
        - If Web Search Results are present, use them to answer the user's question directly.
        - Do NOT say "As an AI" or "I cannot access real-time data." You HAVE the data now.
        - Prioritize the USER'S LAST MESSAGE.
        - Be casual, concise, and friendly.
        <|eot_id|>
        """
        
        # --- 4. GENERATE RESPONSE ---
        print(Fore.GREEN + f"{config.AI_NAME}: ", end="", flush=True)
        
        full_response = ""
        
        # Prepare the message list for Ollama
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.extend(chat_history) # Add recent chat history
        messages.append({'role': 'user', 'content': user_input})

        stream = ollama.chat(
            model=config.MODEL_NAME,
            messages=messages,
            stream=True,
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end="", flush=True)
            full_response += content
            
        print() 
        
        # Update short-term chat history
        chat_history.append({'role': 'user', 'content': user_input})
        chat_history.append({'role': 'assistant', 'content': full_response})
        
        # Keep history short (sliding window)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

if __name__ == "__main__":
    main()