import streamlit as st
import pickle
import torch
import numpy as np
import pandas as pd
import requests
import cv2
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain.schema import Document
from langchain.prompts import PromptTemplate

HF_TOKEN = "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"  # <-- replace with your actual token


# ========================
# UTILITY FUNCTIONS
# ========================

def analyze_image_features(image):
    """Extract visual features and characteristics from an image"""
    try:
        # Convert PIL to OpenCV format for additional analysis
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Basic image properties
        height, width = img_cv.shape[:2]
        
        # Color analysis
        avg_color = np.mean(img_cv, axis=(0, 1))
        dominant_color = "Blue" if avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2] else \
                        "Green" if avg_color[1] > avg_color[2] else "Red"
        
        # Brightness analysis
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_level = "Bright" if brightness > 127 else "Dark"
        
        features = {
            "dimensions": f"{width}x{height}",
            "dominant_color": dominant_color,
            "brightness": brightness_level,
            "aspect_ratio": round(width/height, 2)
        }
        
        return features
    except Exception as e:
        st.error(f"Error analyzing image: {e}")
        return {}

def download_image(url):
    """Download and return PIL Image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Failed to fetch image from {url}: {e}")
        return None

# ========================
# PROMPT TEMPLATES
# ========================

prompt_templates = {
    "image_identification": PromptTemplate.from_template("""
You are a knowledgeable product identification assistant. A user has uploaded an image, and your task is to help them identify the product by analyzing its visual features and comparing it with similar products.

**Visual Features Extracted:**
{visual_features}

**Similar Products in Database:**
{context}

**User Question:**
{question}

Please respond with:
1. Likely product name or category
2. Distinct visual traits you observed
3. Similar or alternative product suggestions
4. Estimated price range based on matches
5. Where the user can find or purchase similar items

Answer:
"""),

    "visual_search": PromptTemplate.from_template("""
You are a smart visual shopping assistant. A user uploaded an image and is seeking product recommendations. Analyze the visual input and match it to similar products in the catalog.

**Extracted Visual Cues:**
{visual_features}

**Similar Items Found:**
{context}

**User Request:**
{question}

Your response should include:
1. Top 3 most visually similar products, with brief justifications
2. Matching visual or design elements
3. Price comparisons between these options
4. Alternative products if the exact match isn't available

Suggested Products:
"""),

    "product_inquiry": PromptTemplate.from_template("""
You are a trusted e-commerce assistant. Use the information below to help the user understand more about a product or make a better purchase decision.

**Product Details Available:**
{context}

**Customer's Question:**
{question}

Respond with:
1. A direct and helpful answer
2. Relevant features and specifications
3. Any visual characteristics that support the answer
4. Related suggestions or alternatives if appropriate

Answer:
""")
}

# ========================
# CHATBOT CLASS
# ========================

class EnhancedMultimodalRAGChatbot:
    def __init__(self, vectorstore, llm_pipeline, clip_model, clip_processor, device):
        self.vectorstore = vectorstore
        self.llm_pipeline = llm_pipeline
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

    def generate_query_embedding(self, query_text=None, query_image=None):
        """Generate embedding for text or image query"""
        try:
            if query_text and query_image:
                # Combined text and image query
                text_inputs = self.clip_processor(text=[query_text], return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)

                image_inputs = self.clip_processor(images=query_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**image_inputs)

                combined = text_features + image_features
                embedding = combined / combined.norm(p=2, dim=-1, keepdim=True)
                return embedding[0].cpu().numpy()

            elif query_text:
                text_inputs = self.clip_processor(text=[query_text], return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.get_text_features(**text_inputs)
                embedding = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                return embedding[0].cpu().numpy()

            elif query_image:
                image_inputs = self.clip_processor(images=query_image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.get_image_features(**image_inputs)
                embedding = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                return embedding[0].cpu().numpy()

        except Exception as e:
            st.error(f"Error generating query embedding: {e}")
            return None

    def identify_product_from_image(self, image, additional_query=""):
        """Identify a product from an uploaded image"""
        # Generate image features
        visual_features = analyze_image_features(image)

        # Generate embedding for the image
        query_embedding = self.generate_query_embedding(query_image=image)
        if query_embedding is None:
            return "Error processing the image. Please try again."

        # Retrieve similar products
        try:
            results = self.vectorstore.similarity_search_by_vector(query_embedding, k=5)
            context = "\n\n".join([
                f"Product {i+1}:\n{r.page_content}\n"
                f"Visual Features: {r.metadata.get('visual_features', {})}"
                for i, r in enumerate(results)
            ])

            # Format prompt for image identification
            prompt_template = prompt_templates["image_identification"]
            question = additional_query if additional_query else "What product is this? Can you identify it and provide details?"

            formatted_prompt = prompt_template.format(
                context=context,
                question=question,
                visual_features=visual_features
            )

            # Generate response
            if self.llm_pipeline:
                try:
                    response = self.llm_pipeline(
                        formatted_prompt,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                    )[0]['generated_text']
                    response = response[len(formatted_prompt):].strip()
                except:
                    response = f"Image Analysis:\nFeatures: {visual_features}\n\nSimilar Products:\n{context}"
            else:
                response = f"Image Analysis:\nFeatures: {visual_features}\n\nSimilar Products:\n{context}"

            return {
                "response": response,
                "visual_features": visual_features,
                "similar_products": results,
                "confidence": "High" if len(results) > 2 else "Medium"
            }

        except Exception as e:
            st.error(f"Error in product identification: {e}")
            return "Sorry, I couldn't identify the product. Please try with a clearer image."

    def visual_search(self, query_image, query_text=""):
        """Search for products similar to the uploaded image"""
        query_embedding = self.generate_query_embedding(query_text, query_image)
        if query_embedding is None:
            return "Error processing your search. Please try again."

        try:
            results = self.vectorstore.similarity_search_by_vector(query_embedding, k=6)

            # Generate image analysis
            visual_features = analyze_image_features(query_image)

            context = "\n\n".join([
                f"Product {i+1}:\n{r.page_content}\n"
                f"Price: {r.metadata.get('price', 'N/A')}\n"
                f"Brand: {r.metadata.get('brand', 'N/A')}"
                for i, r in enumerate(results)
            ])

            prompt_template = prompt_templates["visual_search"]
            question = query_text if query_text else "Find products similar to this image"

            formatted_prompt = prompt_template.format(
                context=context,
                question=question,
                visual_features=visual_features
            )

            if self.llm_pipeline:
                try:
                    response = self.llm_pipeline(
                        formatted_prompt,
                        max_new_tokens=400,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                    )[0]['generated_text']
                    response = response[len(formatted_prompt):].strip()
                except:
                    response = f"Visual Search Results:\n{context}"
            else:
                response = f"Visual Search Results:\n{context}"

            return {
                "response": response,
                "search_results": results,
                "query_analysis": {"features": visual_features}
            }

        except Exception as e:
            st.error(f"Error in visual search: {e}")
            return "Sorry, visual search encountered an error."

    def chat(self, query_text=None, query_image=None, query_type="general", k=3):
        """Enhanced chat function with computer vision capabilities"""
        if not query_text and not query_image:
            return "Please provide either a text query, an image, or both."

        # Handle different query types
        if query_image and query_type == "identify":
            return self.identify_product_from_image(query_image, query_text or "")

        elif query_image and query_type == "search":
            return self.visual_search(query_image, query_text or "")

        elif query_image and query_type == "general":
            # Determine intent based on query text
            if query_text:
                if any(word in query_text.lower() for word in ["what is", "identify", "recognize", "what product"]):
                    return self.identify_product_from_image(query_image, query_text)
                elif any(word in query_text.lower() for word in ["find", "search", "similar", "like this"]):
                    return self.visual_search(query_image, query_text)
            else:
                # Default to identification for image-only queries
                return self.identify_product_from_image(query_image, "")

        # Handle text-only queries
        query_embedding = self.generate_query_embedding(query_text, query_image)
        if query_embedding is None:
            return "Error processing your query. Please try again."

        try:
            results = self.vectorstore.similarity_search_by_vector(query_embedding, k=k)
            context = "\n\n".join([f"Product {i+1}:\n{r.page_content}" for i, r in enumerate(results)])

            prompt_template = prompt_templates["product_inquiry"]
            formatted_prompt = prompt_template.format(context=context, question=query_text)

            if self.llm_pipeline:
                try:
                    response = self.llm_pipeline(
                        formatted_prompt,
                        max_new_tokens=350,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.llm_pipeline.tokenizer.eos_token_id
                    )[0]['generated_text']
                    response = response[len(formatted_prompt):].strip()
                except:
                    response = f"Retrieved products:\n{context}"
            else:
                response = f"Retrieved products:\n{context}"

            return {
                "response": response,
                "retrieved_products": results,
                "num_results": len(results)
            }

        except Exception as e:
            st.error(f"Error in chat function: {e}")
            return "Sorry, I encountered an error processing your request."

# ========================
# STREAMLIT APP
# ========================

@st.cache_resource
def load_models_and_data():
    """Load all models and data (cached for efficiency)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load CLIP
        with st.spinner("Loading CLIP model..."):
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Try to load LLM (optional - app will work without it)
        llm_pipeline = None
        try:
            with st.spinner("Loading language model..."):
                llama_tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct", use_auth_token=HF_TOKEN
                )
                if llama_tokenizer.pad_token is None:
                    llama_tokenizer.pad_token = llama_tokenizer.eos_token
                
                llama_model = AutoModelForCausalLM.from_pretrained(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    device_map="auto",
                    torch_dtype=torch.float16,
                    attn_implementation="eager",
                    use_auth_token=HF_TOKEN
                )
                
                llm_pipeline = pipeline(
                    "text-generation",
                    model=llama_model,
                    tokenizer=llama_tokenizer,
                    device=0 if torch.cuda.is_available() else -1
                )
        except Exception as e:
            st.warning(f"Could not load LLM (will use basic responses): {e}")

        
        # Load saved data - UPDATE THESE PATHS TO YOUR ACTUAL FILE LOCATIONS
        data_paths = {
            "embeddings": "embeddings.pkl",  # Update this path
            "docs": "docs.pkl",              # Update this path
        }
        
        # Try different possible paths
        possible_base_paths = [
            "",  # Current directory
            "./",
            "/content/drive/MyDrive/genaifinal/",  # Your Google Drive path
            "/Users/amytramontozzi/Downloads/",    # Your local path
        ]
        
        embeddings, docs = None, None
        for base_path in possible_base_paths:
            try:
                with open(f"{base_path}embeddings.pkl", "rb") as f:
                    embeddings = pickle.load(f)
                with open(f"{base_path}docs.pkl", "rb") as f:
                    docs = pickle.load(f)
                st.success(f"Data loaded from: {base_path}")
                break
            except FileNotFoundError:
                continue
        
        if embeddings is None or docs is None:
            st.error("Could not find embeddings.pkl and docs.pkl files. Please ensure they are in the same directory as this script or update the paths in the code.")
            return None, None, None, None
        
        # Rebuild FAISS vectorstore
        with st.spinner("Building search index..."):
            dimension = embeddings[0].shape[0]
            index = faiss.IndexFlatIP(dimension)
            index.add(np.stack(embeddings))
            
            docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
            vectorstore = FAISS(
                index=index,
                docstore=docstore,
                index_to_docstore_id={i: str(i) for i in range(len(docs))},
                embedding_function=None
            )
        
        # Initialize chatbot
        chatbot = EnhancedMultimodalRAGChatbot(vectorstore, llm_pipeline, clip_model, clip_processor, device)
        
        return chatbot, clip_model, clip_processor, llm_pipeline
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def main():
    st.set_page_config(
        page_title="🛒 Multimodal Product Assistant",
        page_icon="🛒",
        layout="wide"
    )
    
    st.title("🛒 Multimodal Product Shopping Assistant")
    st.markdown("*Ask questions about products using text, images, or both!*")
    
    # Load models (cached)
    chatbot, clip_model, clip_processor, llm_pipeline = load_models_and_data()
    
    if chatbot is None:
        st.error("Failed to load models. Please check your setup and file paths.")
        st.info("**Setup Instructions:**\n1. Make sure `embeddings.pkl` and `docs.pkl` are in the same directory as this script\n2. Update the file paths in the `load_models_and_data()` function if needed")
        return
    
    # Sidebar for query type selection
    st.sidebar.header("🎯 Query Options")
    query_type = st.sidebar.selectbox(
        "Select query type:",
        ["general", "identify", "search"],
        help="• General: Normal chat\n• Identify: What is this product?\n• Search: Find similar products"
    )
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Text input
        user_text = st.text_area(
            "Enter your question:",
            placeholder="e.g., 'Show me wireless headphones under $100' or 'What is this product?'",
            height=100
        )
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image (optional):",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a product image for identification or visual search"
        )
        
        # Display uploaded image
        query_image = None
        if uploaded_image:
            query_image = Image.open(uploaded_image).convert("RGB")
            st.image(query_image, caption="Uploaded Image", width=300)
    
    with col2:
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        **Text Only:**
        - Ask product questions
        - Request recommendations
        
        **Image Only:**
        - Upload to identify product
        - Find similar items
        
        **Text + Image:**
        - Enhanced search precision
        - Detailed comparisons
        """)
    
    # Process query
    if st.button("🚀 Ask Assistant", type="primary"):
        if not user_text and not uploaded_image:
            st.warning("Please provide either text input or upload an image.")
            return
        
        with st.spinner("🔍 Processing your request..."):
            try:
                # Get response from chatbot
                result = chatbot.chat(
                    query_text=user_text if user_text else None,
                    query_image=query_image,
                    query_type=query_type,
                    k=5
                )
                
                # Display results
                st.subheader("🤖 Assistant Response")
                
                if isinstance(result, dict):
                    # Enhanced response with metadata
                    st.markdown(result.get("response", str(result)))
                    
                    # Show additional info if available
                    if "visual_features" in result:
                        with st.expander("🔍 Visual Analysis"):
                            st.json(result["visual_features"])
                    
                    if "confidence" in result:
                        confidence = result["confidence"]
                        color = "green" if confidence == "High" else "orange" if confidence == "Medium" else "red"
                        st.markdown(f"**Confidence:** :{color}[{confidence}]")
                    
                    # Display similar products
                    products = result.get("similar_products") or result.get("retrieved_products", [])
                    if products:
                        st.subheader("📋 Related Products")
                        for i, product in enumerate(products[:3]):  # Show top 3
                            with st.expander(f"Product {i+1}"):
                                st.write(product.page_content[:300] + "...")
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Price:** {product.metadata.get('price', 'N/A')}")
                                with col_b:
                                    st.write(f"**Brand:** {product.metadata.get('brand', 'N/A')}")
                else:
                    # Simple string response
                    st.markdown(str(result))
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.info("Try with a simpler query or check if your models are loaded correctly.")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Status:**")
    st.sidebar.success("✅ CLIP Model Loaded")
    if llm_pipeline:
        st.sidebar.success("✅ LLM Loaded")
    else:
        st.sidebar.warning("⚠️ LLM Not Available")

if __name__ == "__main__":
    main()