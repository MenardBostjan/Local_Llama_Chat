from llama_cpp import Llama

# --- Model Initialization ---
# Define the path to the language model file.
MODEL_PATH = "C:/Users/menar/.cache/lm-studio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"
# VERY IMPORTANT: DOUBLE-CHECK THIS PATH to ensure it is correct for your system.

# Initialize the Llama model with specified parameters.
llm = Llama(
    model_path=MODEL_PATH,  # Path to the model file.
    n_ctx=131072,         # Context window size (maximum tokens to consider).
    n_threads=12,          # Number of CPU threads to use for inference.
    n_gpu_layers=-1,       # Number of layers to offload to the GPU (-1 for all layers if possible).
    verbose=False           # Enable verbose output for debugging and model loading info.
)

# Print a confirmation message indicating successful model initialization.
print(f"Model initialized successfully from: {MODEL_PATH}")

# --- Response Cleaning Function ---
def clean_ai_response(ai_response):
    """
    Cleans the AI response by removing "thinking aloud" phrases and structuring it for better readability.

    Args:
        ai_response (str): The raw AI generated response string.

    Returns:
        str: The cleaned and more readable AI response string.
    """
    lines = ai_response.strip().split('\n')
    cleaned_lines = []
    thinking = False  # Flag to track if inside a <think> block (if those existed)
    for line in lines:
        line = line.strip()
        if line == "<think>":
            thinking = True
            continue
        elif line == "</think>":  # If you expect closing tag - though your example didn't have one
            thinking = False
            continue

        # Remove common "thinking aloud" prefixes - adjust this list as needed for your model's output style
        if not thinking and not line.startswith("Okay, so") and not line.startswith("Let me start by thinking") and not line.startswith("But I'm not 100% clear") and not line.startswith("I think") and not line.startswith("I'm trying to understand"):
            cleaned_lines.append(line)

    # Further structuring could be added here - e.g., detect points and format as bullets if needed

    return "\n".join(cleaned_lines).strip()


# --- Conversation History Function (using create_chat_completion) ---
def generate_response_with_history(prompt, history=[]):
    """
    Generates a response from the language model using create_chat_completion,
    incorporating the conversation history to maintain context.

    Args:
        prompt (str): The current user input/prompt.
        history (list): A list of strings representing the conversation history.
                        Each string should be in the format "User: message" or "AI: message".

    Returns:
        str: The generated response from the language model, or None if an error occurs.
    """

    # Format messages for the create_chat_completion function.
    # This involves converting the history list into a list of dictionaries
    # where each dictionary represents a message with a 'role' (user or assistant) and 'content'.
    messages = []
    for turn in history:
        role = "user" if turn.startswith("User:") else "assistant"  # Infer role from prefix "User:" or "AI:"
        content = turn[len(role)+2:].strip()  # Extract message content, remove prefix and leading/trailing whitespace
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})  # Add the current user prompt as the latest message

    # --- Debugging Prints (Optional, but helpful for understanding message format) ---
    print("\n--- Messages sent to create_chat_completion ---")
    print(messages)
    print("--- End of Messages ---")
    # --- End of Debugging Prints ---

    try:
        # Use create_chat_completion to generate the model's response.
        response_data = llm.create_chat_completion(
            messages=messages,        # Pass the formatted messages including history and current prompt.
            temperature=0.7,        # Adjust temperature for creativity (higher = more creative, lower = more deterministic).
            max_tokens=200,         # Limit the maximum number of tokens in the generated response.
            stop=["</s>", "###"]     # Define stop sequences to halt response generation (model-specific).
        )
        # Extract the generated response content from the response data.
        response = response_data['choices'][0]['message']['content']
        return response.strip()  # Return the response with leading/trailing whitespace removed.
    except Exception as e:
        # Handle potential errors during the create_chat_completion process.
        print(f"**Error during create_chat_completion:** {e}")
        return None  # Return None to indicate that a response could not be generated.

# --- Main Conversation Loop ---
conversation_history = []  # Initialize an empty list to store the conversation history.

print("Conversation with LLM (DeepSeek R1 Distill Qwen 7B) using create_chat_completion:\n")
while True:  # Start an infinite loop for continuous conversation until the user quits.
    user_input = input("User: ")  # Get user input from the console.
    if user_input.lower() == "quit":  # Check if the user wants to quit the conversation.
        break  # Exit the loop if the user types "quit".

    # Generate AI response using the function that incorporates history.
    ai_response = generate_response_with_history(user_input, conversation_history)

    if ai_response:  # Check if a valid AI response was generated.
        readable_response = clean_ai_response(ai_response) # Clean the AI response to remove thinking aloud parts
        print(f"AI: {readable_response}")  # Print the cleaned AI's response to the console.

        conversation_history.append(f"User: {user_input}")  # Add the user's input to the conversation history.
        conversation_history.append(f"AI: {ai_response}")    # Add the *original* AI's response to the conversation history (important for context!)

        # --- Optional: Print Current Conversation History for Debugging/Review ---
        print("\n--- Current Conversation History ---")
        print(conversation_history)
        print("--- End of History ---")
        # --- End of History Printing ---
    else:
        print("AI: Sorry, I could not generate a response.")  # Inform the user if no response could be generated.

# --- Basic Model Test (outside conversation loop, using create_chat_completion) ---
print("\n--- Basic Model Test (create_chat_completion) ---")
try:
    # Perform a basic test to check if create_chat_completion is working correctly outside the conversation loop.
    test_response_data = llm.create_chat_completion(
        messages=[{"role": "user", "content": "The capital of Japan is "}],  # Simple prompt for testing.
        max_tokens=50  # Limit tokens for the test response.
    )
    test_response = test_response_data['choices'][0]['message']['content'].strip() # Extract and clean the test response.
    print(f"Basic test response (create_chat_completion): {test_response}") # Print the test response.
except Exception as test_e:
    # Handle potential errors during the basic test.
    print(f"**Error during basic test (create_chat_completion):** {test_e}")
    print("--- End of Basic Test (create_chat_completion) ---")