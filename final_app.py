pip install flask transformers langchain langchain-community faiss-cpu torch torchvision requests pillow

import torch
from transformers import CLIPProcessor, CLIPModel
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import numpy as np
from PIL import Image
import requests
from io import BytesIO

device = "cuda" if torch.cuda.is_available() else "cpu"

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

vectorstore = FAISS.load_local("notebook_faiss_test")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

prompt_template = PromptTemplate.from_template("""
You are a helpful e-commerce assistant. Use the following context from product descriptions to answer the user's question.

Context:
{context}

User's Question:
{question}

Answer (in a helpful, friendly tone):
""")
3. Define Functions for Querying
Put the key functions inside the app to:

Compute embedding from text or image query

Query vectorstore for top results

Generate LLM answer

Example:

python
Copy
Edit
def get_text_embedding(text):
    inputs = clip_processor(text=[text], return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
    return features[0].cpu().numpy()

def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features[0].cpu().numpy()

def query_vectorstore(query_embedding, k=3):
    return vectorstore.similarity_search_by_vector(query_embedding, k=k)

def generate_llm_response(results, question):
    context = "\n\n".join([f"{i+1}. {r.page_content}" for i, r in enumerate(results)])
    prompt = prompt_template.format(context=context, question=question)
    return llm.predict(prompt)
4. Flask App Skeleton
Here’s a minimal Flask app example for text queries (you can extend it to accept image uploads similarly):

python
Copy
Edit
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/query_text', methods=['POST'])
def query_text():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Please provide a question"}), 400

    query_embedding = get_text_embedding(question)
    results = query_vectorstore(query_embedding)
    answer = generate_llm_response(results, question)

    # Optional: add top results metadata (e.g., image urls)
    related_products = [{
        "description": r.page_content,
        "image_url": r.metadata.get("source", None)
    } for r in results]

    return jsonify({
        "answer": answer,
        "related_products": related_products
    })

if __name__ == "__main__":
    app.run(debug=True)
