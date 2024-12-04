import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
# Sentiment-to-color mapping
SENTIMENT_COLOR_MAP = {
    'sentimental': '#FF69B4', 'afraid': '#FFD700', 'proud': '#32CD32',
    'faithful': '#87CEEB', 'terrified': '#FF4500', 'joyful': '#7CFC00',
    'angry': '#FF0000', 'sad': '#4682B4', 'jealous': '#228B22',
    'grateful': '#FFA07A', 'prepared': '#008080', 'embarrassed': '#FF6347',
    'excited': '#7B68EE', 'annoyed': '#FF8C00', 'lonely': '#708090',
    'ashamed': '#8B0000', 'guilty': '#A52A2A', 'surprised': '#6A5ACD',
    'nostalgic': '#BC8F8F', 'confident': '#4682B4', 'furious': '#B22222',
    'disappointed': '#D2691E', 'caring': '#FFB6C1', 'trusting': '#2E8B57',
    'disgusted': '#556B2F', 'anticipating': '#DAA520', 'anxious': '#CD5C5C',
    'hopeful': '#00FA9A', 'content': '#FFE4C4', 'impressed': '#4682B4',
    'apprehensive': '#FF7F50', 'devastated': '#8B0000', 'neutral': '#808080'
}

# Load model and tokenizer
model_name = "Thaiebu/Llama-3.1-8B-Empathetic-responses_unsloth"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(model, tokenizer, text, device="cuda"):
    """
    Generate an empathetic response with controlled output and extract sentiment.
    """
    prompt = f"""
Generate a detailed and empathetic response to the task below.

### Task:
Analyze the provided text for emotions and suggest an empathetic, helpful response.

### Text:
{text.strip()}
### Response:
"""

    # Tokenize the input
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    # Generate the response
    generated_output = model.generate(
        **inputs,
        max_new_tokens=150,
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        do_sample=True
    )

    # Decode and clean the output
    response = tokenizer.decode(generated_output[0], skip_special_tokens=True)

    # Extract the part after "### Response:"
    response_parts = response.split("### Response:")
    final_response = response_parts[-1].strip() if len(response_parts) > 1 else response.strip()

    return final_response

def extract_sentiment(response):
    """
    Extracts the sentiment (e.g., frustrated, sad, happy) from the response.
    Assumes the sentiment is mentioned early in the response.
    """
    keywords = ['sentimental', 'afraid', 'proud', 'faithful', 'terrified',
       'joyful', 'angry', 'sad', 'jealous', 'grateful', 'prepared',
       'embarrassed', 'excited', 'annoyed', 'lonely', 'ashamed', 'guilty',
       'surprised', 'nostalgic', 'confident', 'furious', 'disappointed',
       'caring', 'trusting', 'disgusted', 'anticipating', 'anxious',
       'hopeful', 'content', 'impressed', 'apprehensive', 'devastated']
    for keyword in keywords:
        if keyword in response.lower():
            return keyword.capitalize()
    return "Neutral"

# Streamlit chatbot interface
st.set_page_config(page_title="Empathetic Chatbot", page_icon="🤖")
st.title("Empathetic Chatbot 🤗")
st.write("Chat with an AI that provides empathetic responses based on your input.")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Text input box for user input
with st.form(key="chat_form"):
    user_input = st.text_area("Your message:", placeholder="Type something...")
    submit_button = st.form_submit_button(label="Send")

# Generate response if user submits a message
if submit_button and user_input.strip():
    # Append user message to the conversation history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate the bot's response
    bot_response = generate_response(model, tokenizer, user_input, device)

    # Extract the predicted sentiment
    predicted_sentiment = extract_sentiment(bot_response)
    sentiment_color = SENTIMENT_COLOR_MAP.get(predicted_sentiment.lower(), SENTIMENT_COLOR_MAP['neutral'])

    # Append sentiment and bot response to the conversation history
    st.session_state.messages.append({"role": "sentiment", "content": (predicted_sentiment, sentiment_color)})
    st.session_state.messages.append({"role": "bot", "content": bot_response})

# Display chat messages from the conversation history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "sentiment":
        sentiment, color = message["content"]
        st.markdown(f"**Predicted Sentiment:** <span style='color:{color}; font-size:18px;'><b>{sentiment}</b></span>", unsafe_allow_html=True)
    elif message["role"] == "bot":
        st.markdown(f"**Bot:** {message['content']}")