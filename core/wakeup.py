import datetime
import ollama
from colorama import Fore, Style
import config
from memory.episodic import EpisodicMemory

def run_wakeup_routine():
    """
    The Morning Routine.
    1. Checks Date/Time.
    2. Reads the last 'Sleep Summary' from long-term memory.
    3. Generates a proactive greeting based on context.
    """
    print(Style.DIM + "\n[System] Initiating Wake-Up Sequence...")
    
    # 1. Temporal Grounding
    now = datetime.datetime.now()
    current_time = now.strftime("%I:%M %p")
    current_date = now.strftime("%A, %B %d, %Y")
    
    time_context = "Morning"
    if now.hour > 12: time_context = "Afternoon"
    if now.hour > 17: time_context = "Evening"
    
    print(Style.DIM + f"  → Temporal Sync: {current_date} ({current_time})")

    # 2. Memory Retrieval (The "Dream" Residue)
    brain = EpisodicMemory()
    
    # We want the MOST RECENT memory that was formed by the Sleep Cycle
    # We query for "Session Summary" which is the text we used in sleep.py
    recents = brain.recall("Session Summary", n_results=1)
    
    last_context = "You have no memory of previous conversations."
    if recents:
        last_context = f"LAST CONVERSATION SUMMARY: {recents[0]['text']}"
        print(Fore.CYAN + f"  → Recalled Context: {recents[0]['text'][:50]}...")

    # 3. Generate the First Thought
    # We ask Hermes to generate a greeting based on the time and the memory.
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are {config.AI_NAME}. You are waking up.
    
    CURRENT TIME: {current_date} ({time_context})
    {last_context}
    
    INSTRUCTIONS:
    - Generate a proactive opening message to {config.USER_NAME}.
    - If the Last Conversation Summary is relevant, reference it (e.g., "How is that jersey?").
    - If no memory, just say hello and comment on the time of day.
    - Be casual and brief. Do NOT ask "How can I help?".
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0.7} # Slight creativity for the greeting
    )
    
    greeting = response['message']['content']
    return greeting