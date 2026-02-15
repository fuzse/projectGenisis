import ollama
from colorama import Fore, Style, init
import config
from memory.episodic import EpisodicMemory
from memory.sleep import go_to_sleep

# Initialize colors
init(autoreset=True)

def main():
    print(Fore.CYAN + f"Initializing {config.AI_NAME} on {config.MODEL_NAME}...")
    
    # 1. Load Memory
    brain = EpisodicMemory()
    print(Fore.GREEN + f"✔ Hippocampus Linked ({brain.count()} memories)")

    print(Fore.YELLOW + "System Online. (Type 'exit' to quit)")
    
    # Context buffer (Short-term memory)
    # We keep the last 5 turns so it can hold a conversation
    chat_history = [] 

    while True:
        user_input = input(Fore.WHITE + f"\n{config.USER_NAME}: ")
        
        if user_input.lower() in ["exit", "quit"]:
            # Trigger the Sleep Cycle before shutting down
            go_to_sleep(chat_history)
            print(Fore.CYAN + "Shutting down...")
            break
            
        # --- THE COGNITIVE STEP ---
        
        # 1. Recall: "Have I heard about this before?"
        # We search for memories related to the user's input
        memories = brain.recall(user_input, n_results=2)
        
        # Format memories into a string for the prompt
        memory_context = ""
        if memories:
            memory_context = "RELEVANT MEMORIES:\n"
            for m in memories:
                memory_context += f"- {m['text']}\n"
        
        # 2. Construct the Prompt
        system_prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are {config.AI_NAME}. You are chatting with {config.USER_NAME}.
        
        CORE MEMORY (What you know about the user):
        {memory_context if memory_context else "No relevant memories found."}
        
        PERSONALITY RULES:
        1. Speak casually and briefly (1-2 sentences max).
        2. Do NOT act like a customer service agent.
        3. Do NOT ask "How can I help you?".
        4. If the user makes a statement (e.g., "I bought a jersey"), REACT to it. Do not ignore it.
        5. IGNORE previous conversation failures. Focus on the NOW.
        <|eot_id|>
        """
        
        # 3. Generate Response
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
        
        # 4. Consolidate (Save the interaction)
        # In a real brain, we'd sleep on this. For now, we save raw text.
        # We save "User said X, I replied Y" as one memory unit.
        #brain.add_memory(
        #    text=f"User: {user_input} | AI: {full_response}",
        #    metadata={"type": "conversation", "speaker": config.USER_NAME}
        #)
        
        # Update short-term chat history
        chat_history.append({'role': 'user', 'content': user_input})
        chat_history.append({'role': 'assistant', 'content': full_response})
        
        # Keep history short (sliding window)
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]

if __name__ == "__main__":
    main()