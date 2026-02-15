import ollama
from colorama import Fore, Style
import config
from memory.episodic import EpisodicMemory

def go_to_sleep(chat_history):
    if len(chat_history) < 2:
        print(Fore.YELLOW + "Short conversation. No memories to consolidate.")
        return

    print(Style.DIM + "\n[System] Entering Sleep Cycle...")
    print(Style.DIM + "  → Analyzing conversation for key facts...")

    # 1. Format the conversation
    conversation_text = ""
    for msg in chat_history:
        role = "User" if msg['role'] == 'user' else config.AI_NAME
        conversation_text += f"{role}: {msg['content']}\n"

    # 2. The "Few-Shot" Prompt (The Nuclear Fix)
    # We give it examples of exactly what we want.
    extraction_prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a Fact Extraction Engine. You are NOT a chatbot. 
    Your only job is to summarize USER facts from logs.
    
    EXAMPLES:
    Input: 
    User: I like pizza.
    AI: Me too!
    Output: - User likes pizza.

    Input: 
    User: Hi how are you?
    AI: I am good.
    Output: NO_FACTS
    
    Input:
    User: I just bought a Honda Civic.
    AI: Nice car!
    User: Thanks, it arrives tomorrow.
    Output: 
    - User bought a Honda Civic.
    - Car arrives tomorrow.
    
    END EXAMPLES.
    
    Now extract facts from this log:
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {conversation_text}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    # 3. Run with Temperature 0 (Robotic Precision)
    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[{'role': 'user', 'content': extraction_prompt}],
        options={'temperature': 0} 
    )
    
    facts = response['message']['content'].strip()

    # 4. Save
    if "NO_FACTS" in facts or "I'm happy to help" in facts:
        print(Fore.YELLOW + "  → No new facts found.")
    else:
        print(Fore.CYAN + f"  → Extracted Memories:\n{facts}")
        brain = EpisodicMemory()
        brain.add_memory(
            text=f"Session Summary: {facts}",
            metadata={"type": "consolidated_memory", "source": "sleep_cycle"}
        )
        print(Fore.GREEN + "  ✔ Long-term memory updated.")