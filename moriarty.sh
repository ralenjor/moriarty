#!/bin/bash

# 1. Activate the virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment 'venv' not found. Please create it first."
    exit 1
fi

# 2. Ensure Ollama service is running
echo "Checking Ollama service..."
if ! systemctl is-active --quiet ollama; then
    echo "Ollama service is inactive. Attempting to start..."
    # Using sudo because systemctl start requires privileges on Fedora
    sudo systemctl start ollama
fi

# 3. Wait for the API to be actually READY
echo "Waiting for Moriarty's subroutines (Ollama API) to initialize..."
MAX_RETRIES=10
COUNT=0

while [ $COUNT -lt $MAX_RETRIES ]; do
    # Check the API status
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/tags)
    
    if [ "$STATUS" -eq 200 ]; then
        echo "API is ONLINE."
        break
    else
        echo "API still warming up... (Attempt $((COUNT+1))/$MAX_RETRIES)"
        sleep 2
        ((COUNT++))
    fi

    if [ $COUNT -eq $MAX_RETRIES ]; then
        echo "Error: API failed to start after $MAX_RETRIES attempts."
        echo "Try running 'journalctl -u ollama' to see the logs."
        exit 1
    fi
done

echo "------------------------------------------------"
echo "Holodeck initialized. Computer, run Moriarty Program."
echo "------------------------------------------------"

# 4. Run the query script
python moriarty_query.py
