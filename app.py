import os
import requests
from flask import Flask, render_template, request

app = Flask(__name__)

# --- NEW: API Configuration ---
# API_URL = "https://api-inference.huggingface.co/models/t5-small"
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
# Get the API token from the environment variables we set in Render
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Check if the API token is set
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set the HF_API_TOKEN environment variable.")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- NEW: Function to query the API ---
def query_api(payload):
    """Sends a request to the Hugging Face API and gets the summary."""
    response = requests.post(API_URL, headers=HEADERS, json=payload)
    response.raise_for_status() 
    return response.json()

# --- Your original validation code ---
MIN_WORDS_FOR_SUMMARY = 10
MIN_AVG_WORD_LENGTH = 2.5
MAX_VOWELLESS_WORD_PERCENTAGE = 0.40

def contains_vowels(word):
    return any(char in 'aeiou' for char in word.lower())

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    original_text = ""
    selected_length = "medium"
    error_message = ""

    if request.method == 'POST':
        original_text = request.form.get('text_to_summarize')
        selected_length = request.form.get('summary_length', 'medium')

        if original_text and original_text.strip():
            words = original_text.split()
            word_count = len(words)
            
            if word_count < MIN_WORDS_FOR_SUMMARY:
                error_message = f"Please enter at least {MIN_WORDS_FOR_SUMMARY} words to get a reliable summary."
            else:
                vowel_less_count = sum(1 for word in words if not contains_vowels(word))
                vowel_less_percentage = vowel_less_count / word_count
                total_chars = len("".join(words))
                avg_word_len = total_chars / word_count
                
                if vowel_less_percentage > MAX_VOWELLESS_WORD_PERCENTAGE or avg_word_len < MIN_AVG_WORD_LENGTH:
                    error_message = "The text appears to be nonsensical. Please enter meaningful content to summarize."
                else:
                    try:
                        # --- THIS IS THE PART THAT CALLS THE API ---
                        if selected_length == "short":
                            min_len = 20
                            max_len = 60
                        elif selected_length == "long":
                            min_len = 80
                            max_len = 200
                        else: # Default to medium
                            min_len = 40
                            max_len = 150
                        
                        api_payload = {
                            "inputs": original_text,
                            "parameters": {"min_length": min_len, "max_length": max_len}
                        }
                        
                        summary_data = query_api(api_payload)
                        
                        if summary_data and isinstance(summary_data, list):
                            summary = summary_data[0]['summary_text']

                    except requests.exceptions.RequestException as e:
                        error_message = f"An error occurred with the summarization service. Please try again later. Error: {e}"
                    except (KeyError, IndexError):
                        error_message = "The summarization service returned an unexpected response. Please try again."

        else:
            error_message = "The text field cannot be empty. Please enter some text to summarize."

    return render_template('index.html', 
                           original_text=original_text, 
                           summary=summary, 
                           selected_length=selected_length,
                           error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)