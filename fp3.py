import streamlit as st
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image


# Set page configuration
st.set_page_config(page_title="Multitask NLP & Image Tool", layout="centered")
st.title("Multifunctional NLP and Image Generation Tool using Hugging Face Models")

# Load NLP models
# This function loads pretrained NLP models using Hugging Face's pipeline().
# decorator caches the models in memory to avoid reloading them every time the user interacts with the app.
@st.cache_resource
def load_models():
    models = {
        "summarization": pipeline("summarization"),
        "text-generation": pipeline("text-generation", model="gpt2"),
        "chatbot": pipeline("text-generation", model="microsoft/DialoGPT-small"),
        "sentiment-analysis": pipeline("sentiment-analysis"),
        "question-answering": pipeline("question-answering"),
    }
    return models

# Load image generation model
@st.cache_resource
def load_image_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        safety_checker=None  # Disable NSFW filter (use with caution)
    ).to("cuda" if torch.cuda.is_available() else "cpu")  # Auto switch between GPU/CPU
    return pipe

models = load_models()
image_pipe = load_image_model()

# Sidebar for task selection
task = st.sidebar.selectbox(
    "Select Task",
    [
        "Text Summarization",
        "Story Prediction",
        "Next Word Prediction",
        "Chatbot",
        "Sentiment Analysis",
        "Question Answering",
        "Image Generation"
    ]
)

st.markdown("---")

# Task: Text Summarization
if task == "Text Summarization":
    text = st.text_area("Enter text to summarize:")
    if st.button("Summarize") and text:
        summary = models["summarization"](text, max_length=130, min_length=30, do_sample=False)
        st.success(summary[0]['summary_text'])

# Task: Story Prediction
elif task == "Story Prediction":
    prompt = st.text_area("Enter story beginning:")
    if st.button("Generate Story") and prompt:
        output = models["text-generation"](prompt, max_length=100, num_return_sequences=1)
        st.success(output[0]['generated_text'])

# Task: Next Word Prediction
elif task == "Next Word Prediction":
    prompt = st.text_input("Start your sentence:")
    if st.button("Predict") and prompt:
        set_seed(42)
        output = models["text-generation"](prompt, max_length=len(prompt.split()) + 10, num_return_sequences=1)
        st.success(output[0]['generated_text'])


# Task: Chatbot
elif task == "Chatbot":
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None
    if "bot_input_ids" not in st.session_state:
        st.session_state.bot_input_ids = None

    user_input = st.text_input("You:")
    if st.button("Respond") and user_input:

        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

        # Encode user input + history
        new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
        bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids

        # Generate response
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode and show
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        st.success("Bot: " + response)

        # Save for next round
        st.session_state.chat_history_ids = chat_history_ids


# Task: Sentiment Analysis
elif task == "Sentiment Analysis":
    text = st.text_input("Enter text to analyze:")
    if st.button("Analyze") and text:
        result = models["sentiment-analysis"](text)
        label, score = result[0]['label'], result[0]['score']
        st.success(f"Sentiment: {label} (Confidence: {score:.2f})")

# Task: Question Answering
elif task == "Question Answering":
    context = st.text_area("Enter context:")
    question = st.text_input("Ask a question:")
    if st.button("Answer") and context and question:
        result = models["question-answering"](question=question, context=context)
        st.success(f"Answer: {result['answer']}")

# Task: Image Generation
elif task == "Image Generation":
    prompt = st.text_input("Enter image prompt:", placeholder="e.g. A futuristic cityscape at sunset")

    if st.button("Generate Image") and prompt:
        try:
            with torch.no_grad():
                result = image_pipe(prompt)
                image = result.images[0]

            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)

            # Optional: Save to file
            image.save("generated_image.png")
            st.success("Image saved as generated_image.png")

        except Exception as e:
            st.error(f"Image generation failed: {str(e)}")
