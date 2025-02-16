from llama_cpp import Llama
import os

# Configuration
MODEL_PATH = "C:/Users/menar/.cache/lm-studio/models/lmstudio-community/DeepSeek-R1-Distill-Qwen-7B-GGUF/DeepSeek-R1-Distill-Qwen-7B-Q6_K.gguf"  # ← Update path
SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely and clearly."

# Check model exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}")
    exit()

# Initialize model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=12,
    n_gpu_layers=-1,  # Set based on GPU capability (0 for CPU-only)
    offload_kqv=True,
    verbose=True
)

print("Chat initialized. Type 'exit' to quit.\n")

# Chat loop
while True:
    try:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
            
        # Create message history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ]
        
        # Generate response
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            stop=["</s>", "###"]  # Stop sequences
        )
        
        # Print response
        print("\nAssistant:", response['choices'][0]['message']['content'])
        print("")  # Empty line for spacing
        
    except KeyboardInterrupt:
        break

print("\nChat session ended.")


