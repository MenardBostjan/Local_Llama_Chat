from llama_cpp import Llama

# --- Model Initialization ---
MODEL_PATH = "C:/Users/menar/.cache/lm-studio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"  # <---  VERY IMPORTANT: DOUBLE-CHECK THIS PATH!

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=131072,
    n_threads=12,
    n_gpu_layers=-1,
    verbose=True
)

print(f"Model initialized successfully from: {MODEL_PATH}")

# --- Conversation History Function (using create_chat_completion) ---
def generate_response_with_history(prompt, history=[]):
    """
    Generates a response using llm.create_chat_completion, incorporating history.
    """

    # Format messages for create_chat_completion
    messages = []
    for turn in history:
        role = "user" if turn.startswith("User:") else "assistant" # Infer role from prefix
        content = turn[len(role)+2:].strip() # Remove "User: " or "AI: " prefix and strip whitespace
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt}) # Add the current user prompt

    print("\n--- Messages sent to create_chat_completion ---") # Debugging print
    print(messages)                                         # Debugging print
    print("--- End of Messages ---")                        # Debugging print


    try:
        response_data = llm.create_chat_completion( # Using create_chat_completion now!
            messages=messages,
            temperature=0.7,      # Adjust as needed
            max_tokens=200,       # Adjust as needed
            stop=["</s>", "###"]   # Stop sequences (check model documentation)
        )
        response = response_data['choices'][0]['message']['content'] # Extract response content
        return response.strip()
    except Exception as e:
        print(f"**Error during create_chat_completion:** {e}")
        return None

# --- Main Conversation Loop ---
conversation_history = []

print("Conversation with LLM (DeepSeek R1 Distill Qwen 7B) using create_chat_completion:\n")
while True:
    user_input = input("User: ")
    if user_input.lower() == "quit":
        break

    ai_response = generate_response_with_history(user_input, conversation_history)

    if ai_response:
        print(f"AI: {ai_response}")
        conversation_history.append(f"User: {user_input}") # Still store history as "User: ..." and "AI: ..."
        conversation_history.append(f"AI: {ai_response}")
        print("\n--- Current Conversation History ---")
        print(conversation_history)
        print("--- End of History ---")
    else:
        print("AI: Sorry, I could not generate a response.")

# --- Basic Model Test (outside conversation loop, using create_chat_completion) ---
print("\n--- Basic Model Test (create_chat_completion) ---")
try: 
    test_response_data = llm.create_chat_completion(
        messages=[{"role": "user", "content": "The capital of Japan is "}], # Message format
        max_tokens=50)
    test_response = test_response_data['choices'][0]['message']['content'].strip()
    print(f"Basic test response (create_chat_completion): {test_response}")
except Exception as test_e:
    print(f"**Error during basic test (create_chat_completion):** {test_e}")
    print("--- End of Basic Test (create_chat_completion) ---")