from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load the smaller, faster model
# summarizer = pipeline("summarization", model="ssleifer/distilbart-cnn-12-6")
summarizer = pipeline("summarization", model="t5-small")

# --- NEW ---: More advanced validation constants
MIN_WORDS_FOR_SUMMARY = 10
MIN_AVG_WORD_LENGTH = 2.5  # We can lower this slightly as the vowel check is more powerful
MAX_VOWELLESS_WORD_PERCENTAGE = 0.40 # Allow up to 40% of words to have no vowels (e.g., "rhythm", "Mr.", etc.)

# --- NEW ---: Helper function to check for vowels in a word
def contains_vowels(word):
    """Checks if a word contains at least one vowel (a, e, i, o, u)."""
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
            
            # 1. First check: Minimum word count
            if word_count < MIN_WORDS_FOR_SUMMARY:
                error_message = f"Please enter at least {MIN_WORDS_FOR_SUMMARY} words to get a reliable summary."
            else:
                # --- NEW ---: Advanced checks for nonsensical content
                
                # 2. Second check: Vowel presence
                vowel_less_count = sum(1 for word in words if not contains_vowels(word))
                vowel_less_percentage = vowel_less_count / word_count
                
                # 3. Third check: Average word length
                total_chars = len("".join(words))
                avg_word_len = total_chars / word_count
                
                if vowel_less_percentage > MAX_VOWELLESS_WORD_PERCENTAGE or avg_word_len < MIN_AVG_WORD_LENGTH:
                    error_message = "The text appears to be nonsensical. Please enter meaningful content to summarize."
                else:
                    # ALL CHECKS PASSED! Proceed with summarization.
                    if selected_length == "short":
                        min_len = 20
                        max_len = 60
                    elif selected_length == "long":
                        min_len = 80
                        max_len = 200
                    else: # Default to medium
                        min_len = 40
                        max_len = 150
                    
                    summary_list = summarizer(original_text, max_length=max_len, min_length=min_len, do_sample=False)
                    summary = summary_list[0]['summary_text']
        else:
            # Handle empty input
            error_message = "The text field cannot be empty. Please enter some text to summarize."

    # Pass the error_message to the template
    return render_template('index.html', 
                           original_text=original_text, 
                           summary=summary, 
                           selected_length=selected_length,
                           error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)