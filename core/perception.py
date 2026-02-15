import ollama
import config
from colorama import Fore

def check_intent(user_input):
    """
    Decides if the AI needs external tools to answer.
    Returns: "search: <query>" or "respond"
    """
    prompt = f"""
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are a hidden intent classifier.
    
    TASK: Determine if the USER input requires a Google Search to answer accurately.
    - If yes, output "SEARCH: <optimized search query>"
    - If no (it's casual chat, opinion, or personal), output "RESPOND"
    
    EXAMPLES:
    Input: "Who won the Super Bowl?" -> Output: SEARCH: Super Bowl winner 2026
    Input: "I like pizza." -> Output: RESPOND
    Input: "What is the weather in Maryland?" -> Output: SEARCH: current weather Maryland
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_input}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    response = ollama.chat(
        model=config.MODEL_NAME,
        messages=[{'role': 'user', 'content': prompt}],
        options={'temperature': 0}
    )
    
    result = response['message']['content'].strip()
    return result