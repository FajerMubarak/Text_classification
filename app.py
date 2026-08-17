import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Restaurant Review Analyzer", page_icon="🍔", layout="centered")

st.title("🍔 Arabic Restaurant Reviews Analyzer")
st.write("Enter a review text to instantly get sentiment classification and category detection.")

# 1. Load models with caching for optimal performance and preventing reload
@st.cache_resource
def load_pipelines():
    # Model for Sentiment Classification
    sentiment_pipe = pipeline("text-classification", model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment")
    
    # Model for Zero-Shot Category Classification
    zero_shot_pipe = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    
    return sentiment_pipe, zero_shot_pipe

sentiment_model, zero_shot_model = load_pipelines()

# Approved candidate categories
candidate_categories = ["Food Quality", "Service & Staff", "Price", "Location & Ambience", "Waiting Time"]

# 2. Text input UI
user_review = st.text_area("Review Text:", value="الاكل لذيذ جدا صراحه والمكان نظيف بس الخدمة كانت بطيئة", height=100)

if st.button("Analyze Review 🚀"):
    if user_review.strip() != "":
        with st.spinner("Analyzing text..."):
            # Sentiment prediction
            sent_res = sentiment_model(user_review[:512])[0]
            
            # Category prediction
            cat_res = zero_shot_model(user_review[:512], candidate_labels=candidate_categories)
            
            top_category = cat_res['labels'][0]
            cat_score = cat_res['scores'][0]

            st.divider()

            # Display results in two columns
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🎭 Sentiment")
                st.success(f"**Label:** {sent_res['label']}")
                st.write(f"Confidence: `{sent_res['score']:.2%}`")

            with col2:
                st.subheader("🏷️ Category")
                st.info(f"**Category:** {top_category}")
                st.write(f"Confidence: `{cat_score:.2%}`")

            # Display custom metadata output
            st.divider()
            st.caption("Custom Metadata Output:")
            st.json({
                "sentiment_output": {
                    "label": sent_res['label'],
                    "score": round(float(sent_res['score']), 4),
                    "metadata": "huggingface_AI_model"
                },
                "category_output": {
                    "label": top_category,
                    "score": round(float(cat_score), 4),
                    "metadata": "huggingface_AI_model"
                }
            })
    else:
        st.warning("Please enter a review before analyzing.")
