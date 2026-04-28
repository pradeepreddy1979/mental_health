# Mental Health Disorder Detection App

This is a Streamlit application that uses AI (via Ollama API) to detect mental health disorders based on symptom descriptions.

## Prerequisites

1. Install Ollama: Download and install from https://ollama.ai/
2. Pull a model, e.g., `ollama pull tinyllama` (smaller model for low memory systems)
3. Start Ollama server: `ollama serve` (runs on localhost:11434)

## Installation

1. Install Python packages:
   ```
   pip install -r requirements.txt
   ```

## Running the App

1. Ensure Ollama is running.
2. Run the app:
   ```
   streamlit run app.py
   ```
3. Open the provided URL in your browser.
4. Enter a description of symptoms in the text area.
5. Click "Detect Disorder" to get the AI's classification.

## Notes

- The app classifies into conditions found in the CSV: anxiety, bipolar, stress, personality disorder.
- If the API key is required for your Ollama setup, it is included in the code (uncomment the header line if needed).
- Make sure to change the model name in the code if using a different one (currently set to tinyllama for memory efficiency).