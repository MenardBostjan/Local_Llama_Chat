from llama_cpp import Llama

# --- Model Initialization ---
MODEL_PATH = "C:/Users/menar/.cache/lm-studio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"  # Replace with your actual path
llm = Llama(
    model_path=MODEL_PATH,
    #n_ctx=130000,
    n_ctx=13000,
    n_threads=10,
    n_gpu_layers=-1,
    verbose=False
)
print(f"Model initialized successfully from: {MODEL_PATH}")

# --- Response Formatting Function (No Thinking Removal) ---
def format_response(ai_response):
    """Formats the AI response, handling lists and other formatting."""

    lines = ai_response.strip().split('\n')
    formatted_lines = []
    in_list = False

    for line in lines:
        line = line.strip()

        # List formatting
        if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
            formatted_lines.append("• " + line[2:].strip())
            in_list = True
        elif in_list:
            if line != "":
                formatted_lines[-1] += " " + line  # Handle multi-line list items
            else:
                in_list = False
                formatted_lines.append("")  # Add a newline to separate lists
        else:
            formatted_lines.append(line)
            in_list = False

    return "\n".join(formatted_lines).strip()


# --- Conversation History Function ---
def generate_response_with_history(prompt, history=[]):
    messages = []
    for turn in history:
        role = "user" if turn.startswith("User:") else "assistant"
        content = turn[len(role) + 2:].strip()
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})

    # --- Debugging Prints (Optional) ---
    # print("\n--- Messages sent to create_chat_completion ---")
    # print(messages)
    # print("--- End of Messages ---")
    # --- End of Debugging Prints ---

    try:
        response_data = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=55000,
            #stop=["</s>", "###"]
            stop=["<|im_end|>"]
        )
        response = response_data['choices'][0]['message']['content']
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
        formatted_response = format_response(ai_response)  # Use the formatting function
        print(f"AI: {formatted_response}")

        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"AI: {ai_response}")

        # --- Optional: Print Current Conversation History ---
        # print("\n--- Current Conversation History ---")
        # print(conversation_history)
        # print("--- End of History ---")
        # --- End of History Printing ---

    else:
        print("AI: Sorry, I could not generate a response.")


# # --- Basic Model Test ---
# print("\n--- Basic Model Test (create_chat_completion) ---")
# try:
    # test_response_data = llm.create_chat_completion(
        # messages=[{"role": "user", "content": "The capital of Japan is "}],
        # max_tokens=500
    # )
    # test_response = test_response_data['choices'][0]['message']['content'].strip()
    # print(f"Basic test response (create_chat_completion): {test_response}")
# except Exception as test_e:
    # print(f"**Error during basic test (create_chat_completion):** {test_e}")
    # print("--- End of Basic Test (create_chat_completion) ---")