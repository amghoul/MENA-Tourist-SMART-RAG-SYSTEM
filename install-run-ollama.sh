#!/bin/bash

# --- 1. System Updates and Dependencies ---
echo "--- 1. Updating system and installing pciutils ---"
sudo apt update
sudo apt install -y pciutils

# --- 2. Ollama Installation ---
echo "--- 2. Installing Ollama ---"
# Download and run the official Ollama installation script
curl -fsSL https://ollama.com/install.sh | sh

# --- 3. Set Environment Variables and Start Server ---
# Set the host to 0.0.0.0 to make it accessible outside of localhost (essential for Colab)
# Use the `export` command to set variables for the subsequent commands
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_ORIGINS="*"

echo "--- 3. Starting Ollama Server in Background ---"
# The 'ollama serve' command starts the server.
# The '&' runs it in the background immediately, allowing the script to continue.
nohup ollama serve > /dev/null 2>&1 &

# Optional: Capture the Process ID (PID) of the background server for later management
SERVER_PID=$!
echo "Ollama server started with PID: $SERVER_PID"

# --- 4. Pull a Model (Wait for server to initialize) ---
# Give the server a moment to boot up before attempting to pull a model.
echo "--- 4. Waiting 5 seconds for server to initialize ---"
sleep 5

echo "--- 5. Pulling the self.llm = OllamaLLM(model="mistral:7b") model ---"
# Pull a model. This command will execute while the server runs in the background.
# You can replace 'llama3' with any model you want (e.g., mistral, llama2).
ollama pull mistral:7b

# --- 6. Confirmation and Next Steps ---
echo "--------------------------------------------------------"
echo "✅ Ollama setup complete!"
echo "Server is running on $OLLAMA_HOST"
echo "To use this server, you will likely need to expose the port (11434) using Ngrok or a similar service."
echo "--------------------------------------------------------"