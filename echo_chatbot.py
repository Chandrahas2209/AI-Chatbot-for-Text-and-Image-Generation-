import streamlit as st
from huggingface_hub import InferenceClient
import requests
from PIL import Image
from io import BytesIO

# Hugging Face API Token
HF_API_TOKEN = ""  
TEXT_MODEL = "microsoft/Phi-3.5-mini-instruct"  # Text generation model

# Clipdrop API Key
CLIPDROP_API_KEY = ""  

# Hugging Face Client
client = InferenceClient(token=HF_API_TOKEN)

# Streamlit App
st.title("AI Chatbot: Text and Image Generation")
st.subheader("Ask me anything or describe an image to generate!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_data"):
            st.image(message["image_data"], caption=message["content"])
        else:
            st.markdown(message["content"])

# Generate text response
def generate_text(prompt):
    try:
        response = client.text_generation(
            model=TEXT_MODEL,
            prompt=prompt,
            max_new_tokens=2200,
            temperature=0.7
        )
        return response if isinstance(response, str) else response[0]["generated_text"]
    except Exception as e:
        return f"Error generating text: {e}"

# Generate image response
def generate_image(prompt):
    try:
        response = requests.post(
            'https://clipdrop-api.co/text-to-image/v1',
            files={'prompt': (None, prompt, 'text/plain')},
            headers={'x-api-key': CLIPDROP_API_KEY}
        )
        if response.ok:
            return Image.open(BytesIO(response.content))
        else:
            st.error(f"Image generation failed: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Chat input handling
if user_input := st.chat_input("Type here..."):
    # Add user input to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a text response
    response = generate_text(user_input)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Check if input mentions generating an image
    if "image" in user_input.lower():
        image = generate_image(user_input)
        if image:
            with st.chat_message("assistant"):
                st.image(image, caption=f"Generated Image for: {user_input}")
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Generated Image for: {user_input}", "image_data": image}
            )
        else:
            error_message = "Sorry, I couldn't generate an image. Try again later."
            with st.chat_message("assistant"):
                st.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
