import streamlit as st
import pickle
import faiss
from huggingface_hub import InferenceClient
from PIL import Image
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


# === CONFIG ===
SAVE_PATH = "/Users/amytramontozzi/Downloads/"
HF_API_KEY = "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"
HF_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

# === LOAD DATA ===
@st.cache_data(show_spinner=True)
def load_data():
    with open(f"{SAVE_PATH}embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open(f"{SAVE_PATH}docs.pkl", "rb") as f:
        docs = pickle.load(f)
    with open(f"{SAVE_PATH}product_images.pkl", "rb") as f:
        product_images = pickle.load(f)
    index = faiss.read_index(f"{SAVE_PATH}faiss.index")
    return embeddings, docs, product_images, index

embeddings, docs, product_images, faiss_index = load_data()

# === LOAD CLIP MODEL & PROCESSOR ===
@st.cache_resource
def load_clip():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return processor, model

processor, model = load_clip()

def embed_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
    return text_features.cpu().numpy()

def embed_images(images):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    return image_features.cpu().numpy()

# === HUGGINGFACE LLM CLIENT ===
try:
    client = InferenceClient(api_key=HF_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize HF client: {e}")
    client = None

def query_llm(prompt: str) -> str:
    # Temporarily bypass HF API to test if that's the issue
    return "API temporarily disabled for testing. Here are the relevant products based on your search."

# === SEARCH FUNCTION ===
def search_similar_docs(query_embedding, k=3):
    D, I = faiss_index.search(query_embedding.astype("float32"), k)
    results = [docs[idx] for idx in I[0]]
    return results

# === BUILD PROMPT WITH CONTEXT ===
def build_prompt(user_query, context_docs):
    context_text = "\n\n---\n\n".join(context_docs)
    prompt = (
        f"You are a helpful product chatbot.\n"
        f"Use the following product information to answer the user's question:\n\n{context_text}\n\n"
        f"User question: {user_query}\n"
        f"Answer in a clear and concise manner."
    )
    return prompt

# === STREAMLIT APP UI ===
st.title("🛍️ Multimodal Product Chatbot")

st.markdown(
    """
    Enter your question or upload an image of a product to get recommendations and answers.
    """
)

query_type = st.radio("Select query type:", ["Text", "Image"])

if query_type == "Text":
    user_input = st.text_input("Type your question here:")
    submitted = st.button("Ask")

    if submitted and user_input:
        st.write("Starting query processing...")  # Debug
        try:
            with st.spinner("Searching and generating response..."):
                st.write("Embedding text...")  # Debug
                q_embedding = embed_text([user_input])
                st.write("Searching documents...")  # Debug
                relevant_docs = search_similar_docs(q_embedding, k=3)
                st.write("Building prompt...")  # Debug
                prompt = build_prompt(user_input, relevant_docs)
                st.write("Calling LLM...")  # Debug
                answer = query_llm(prompt)

            st.markdown("### 🤖 Chatbot Response:")
            st.write(answer)

            st.markdown("### 🖼️ Related Products:")
            for i, doc in enumerate(relevant_docs):
                try:
                    idx = docs.index(doc)
                    img = product_images[idx]
                    if isinstance(img, str):
                        st.image(img, width=150)
                    elif isinstance(img, np.ndarray):
                        st.image(img, width=150)
                    elif isinstance(img, Image.Image):
                        st.image(img, width=150)
                except Exception as img_e:
                    st.write(f"Error loading image {i}: {img_e}")
        except Exception as e:
            st.error(f"Error in main processing: {e}")
            import traceback
            st.text(traceback.format_exc())

elif query_type == "Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Search Similar Products"):
            try:
                with st.spinner("Processing image and searching..."):
                    q_embedding = embed_images([image])
                    relevant_docs = search_similar_docs(q_embedding, k=3)
                    
                    prompt = build_prompt("What products are similar to this image?", relevant_docs)
                    answer = query_llm(prompt)

                st.markdown("### 🤖 Chatbot Response:")
                st.write(answer)

                st.markdown("### 🖼️ Similar Products:")
                for doc in relevant_docs:
                    idx = docs.index(doc)
                    img = product_images[idx]
                    if isinstance(img, str):
                        st.image(img, width=150)
                    elif isinstance(img, np.ndarray):
                        st.image(img, width=150)
                    elif isinstance(img, Image.Image):
                        st.image(img, width=150)
            except Exception as e:
                st.error(f"Error: {e}")