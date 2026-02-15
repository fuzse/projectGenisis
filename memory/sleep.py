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
    
    RULES:
    1. Extract facts about the USER (preferences, life events, location).
    2. IGNORE everything the AI (Genesis) said or did.
    3. Do NOT record the date or time (the system handles that).
    4. If the user says nothing of substance, output "NO_FACTS".
    
    EXAMPLES:
    Input: 
    User: I love the Broncos.
    AI: I like them too.
    Output: - User loves the Broncos.

    Input: 
    User: When is the draft?
    AI: In April.
    Output: NO_FACTS  <-- (Because the User didn't reveal info about themselves)
    
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