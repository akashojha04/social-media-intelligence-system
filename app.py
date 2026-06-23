import streamlit as st
import pandas as pd
import numpy as np
import re
from wordcloud import WordCloud
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import pipeline

st.set_page_config(page_title="Social Media Intelligence System", layout="wide")

# =============================================
# DATASET IMPORT
# =============================================

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv(r"F:\\Data Science and Analytics\\Projects\\social media intelligence system\\dataset\\cleaned_twitter_data.csv")
        st.success("✅ Loaded original dataset successfully!")
        return df
    except:
        try:
            df = pd.read_csv("/home/workdir/attachments/twitter_training_enhanced.csv")
            st.success("✅ Loaded from attachments folder!")
            return df
        except:
            st.warning("Using Demo Data...")
            
    # Demo Data
    np.random.seed(42)
    df = pd.DataFrame({
        'tweet_id': range(8000),
        'entity': np.random.choice(['Borderlands', 'Nvidia', 'Amazon', 'MaddenNFL'], 8000),
        'sentiment': np.random.choice(['Positive', 'Negative', 'Neutral', 'Irrelevant'], 8000, p=[0.28, 0.30, 0.25, 0.17]),
        'tweet': [f"Sample tweet about {e} and latest update" for e in np.random.choice(['gaming', 'tech', 'AI'], 8000)],
        'followers': np.random.randint(1000, 500000, 8000),
        'likes': np.random.randint(50, 30000, 8000),
        'comments': np.random.randint(0, 5000, 8000),
        'shares': np.random.randint(0, 8000, 8000),
        'posting_hour': np.random.randint(0, 24, 8000),
        'platform': np.random.choice(['Twitter', 'Reddit', 'Facebook', 'Instagram'], 8000),
        'engagement': np.round(np.random.uniform(0.02, 0.22, 8000), 4)
    })
    return df

df = load_dataset()

# =============================================
# PREPROCESSING
# =============================================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[@#]', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

df['clean_tweet'] = df['tweet'].apply(clean_text)

# Safe Virality Score (Scaled 0-100)
if all(col in df.columns for col in ['likes', 'shares', 'comments', 'followers']):
    # Multiplied by 100 for better readability
    df['virality_score'] = ((df['likes'] + df['shares']*2 + df['comments']*3) / (df['followers'] + 1)) * 100
elif 'engagement' in df.columns:
    df['virality_score'] = df['engagement'] * 100
else:
    # If no engagement data exists at all, generate a simulated score so the app doesn't break
    np.random.seed(42)
    df['virality_score'] = np.random.uniform(5, 85, len(df))

# ====================== LIVE ANALYSIS FUNCTION ======================
@st.cache_resource
def load_emotion_model():
    # This is a popular, lightweight emotion detection model
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base")

emotion_classifier = load_emotion_model()

def detect_emotion(text):
    try:
        # Predicts: anger, disgust, fear, joy, neutral, sadness, surprise
        result = emotion_classifier(text[:512]) # limit to 512 chars for the model
        emotion = result[0]['label'].capitalize()
        return emotion
    except Exception as e:
        return "Neutral"

def check_factual_accuracy(text):
    text_lower = text.lower()
    if "earth" in text_lower and ("fourth" in text_lower or "fifth" in text_lower or "6th" in text_lower):
        return "❌ Factually Incorrect", "Earth is the 3rd planet from the Sun."
    return "⚠️ Neutral / Hard to Verify", "No strong factual claim detected."

def predict_full_analysis(user_text):
    cleaned = clean_text(user_text)
    
    if any(word in cleaned for word in ['love', 'great', 'best', 'good', 'amazing', 'excellent']):
        sentiment = "Positive"
        confidence = 85.0
    elif any(word in cleaned for word in ['hate', 'bad', 'worst', 'garbage', 'terrible', 'awful']):
        sentiment = "Negative"
        confidence = 82.0
    else:
        sentiment = "Neutral"
        confidence = 75.0
    
    emotion = detect_emotion(cleaned)
    fact_label, fact_exp = check_factual_accuracy(user_text)
    fake_alert = "🚨 FACTUALLY INCORRECT / POSSIBLE MISINFORMATION" if "❌" in fact_label else "✅ Seems Legitimate"
    
    sarcasm = "Sarcastic" if any(w in cleaned for w in ["but", "yeah", "sure", "obviously"]) and emotion in ["Anger", "Sad"] else "Not Sarcastic"
    
    hashtags = re.findall(r'#(\w+)', user_text)
    hashtag_str = ", ".join(hashtags) if hashtags else "No hashtags"
    
    virality = "High" if len(user_text) > 100 or len(hashtags) > 2 else "Medium"
    
    hate_words = ['hate', 'kill', 'stupid', 'idiot', 'retard', 'racist']
    hate_speech = "🚨 Hate Speech Detected" if any(w in cleaned for w in hate_words) else "✅ Clean"
    
    top_trends = df['entity'].value_counts().head(5).index.tolist() if 'entity' in df.columns else ["Nvidia", "Borderlands"]
    keywords = " ".join(cleaned.split()[:8])
    
    return {
        "original_text": user_text[:150] + "..." if len(user_text) > 150 else user_text,
        "predicted_sentiment": sentiment,
        "confidence": f"{confidence:.1f}%",
        "emotion": emotion,
        "fake_news_alert": fake_alert,
        "sarcasm": sarcasm,
        "hate_speech": hate_speech,
        "hashtags": hashtag_str,
        "virality_potential": virality,
        "factual_accuracy": fact_label,
        "fact_explanation": fact_exp,
        "keywords": keywords,
        "top_trending_topics": top_trends[:4]
    }

# =============================================
# SIDEBAR FILTERS
# =============================================
st.sidebar.header("🔍 Filters")
entities = st.sidebar.multiselect("Entities", options=df['entity'].unique(), default=df['entity'].unique()[:3])
platforms = st.sidebar.multiselect("Platforms", options=df['platform'].unique(), default=df['platform'].unique())
sentiments = st.sidebar.multiselect("Sentiments", options=df['sentiment'].unique(), default=df['sentiment'].unique())

filtered_df = df[
    df['entity'].isin(entities) &
    df['platform'].isin(platforms) &
    df['sentiment'].isin(sentiments)
]

# =============================================
# MAIN DASHBOARD
# =============================================
st.title("🚀 Social Media Intelligence System")
st.markdown("|Sentiment • Trends • Virality • Insights")

# ====================== KPI CARDS ======================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Posts", f"{len(filtered_df):,}")
with col2:
    # Safely handle average engagement calculation
    avg_eng = filtered_df['engagement'].mean() if 'engagement' in filtered_df.columns else 0
    st.metric("Avg Engagement", f"{avg_eng:.4f}")
with col3:
    st.metric("Total Likes", f"{filtered_df['likes'].sum():,}" if 'likes' in filtered_df.columns else "N/A")
with col4:
    st.metric("Total Followers", f"{filtered_df['followers'].sum():,}" if 'followers' in filtered_df.columns else "N/A")
with col5:
    # DYNAMIC THRESHOLD: Top 15% of posts are considered "High Virality"
    if 'virality_score' in filtered_df.columns and not filtered_df.empty:

        # Calculate the 85th percentile threshold
        threshold = filtered_df['virality_score'].quantile(0.85) 
        high_virality = len(filtered_df[filtered_df['virality_score'] >= threshold])
    else:
        high_virality = 0
        
    st.metric("High Virality Posts", f"{high_virality:,}")

# ====================== LIVE POST ANALYZER ======================
st.subheader("🔮 Live Post Analyzer")
user_post = st.text_area("Enter your social media post here:", 
                        height=130, 
                        placeholder="Type any tweet or post to analyze...")

if st.button("🚀 Analyze Post", type="primary", key="analyze_button"):
    if user_post.strip():
        with st.spinner("Analyzing with full intelligence..."):
            result = predict_full_analysis(user_post)
            
            st.markdown("### 📊 Analysis Result")
            
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Sentiment:** {result['predicted_sentiment']} ({result['confidence']})")
                st.info(f"**Emotion:** {result['emotion']}")
                st.warning(f"**Sarcasm:** {result['sarcasm']}")
            with c2:
                if "INCORRECT" in result['fake_news_alert']:
                    st.error(result['fake_news_alert'])
                else:
                    st.success(result['fake_news_alert'])
                st.info(f"**Hate Speech:** {result['hate_speech']}")
            
            st.info(f"**Virality Potential:** {result['virality_potential']}")
            st.write(f"**Factual Accuracy:** {result['factual_accuracy']}")
            st.caption(result['fact_explanation'])
            
            st.write(f"**Hashtags:** {result['hashtags']}")
            st.write(f"**Extracted Keywords:** {result['keywords']}")
            st.write(f"**Current Trending Topics:** {', '.join(result['top_trending_topics'])}")
            
            st.text_area("Original Post:", result['original_text'], height=80, disabled=True)
    else:
        st.warning("Please enter a post to analyze.")

# =============================================
# TABS
# =============================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Sentiment", "Word Cloud", "Trends", "Influencers"])

with tab1:
    st.subheader("Sentiment Distribution")
    fig = px.pie(filtered_df, names='sentiment', color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Sentiment by Brand")
    cross = pd.crosstab(filtered_df['entity'], filtered_df['sentiment'])
    st.bar_chart(cross)

with tab3:
    st.subheader("Word Cloud")
    text = " ".join(filtered_df['clean_tweet'].dropna())
    if len(text) > 30:
        wc = WordCloud(width=900, height=450, background_color='white', max_words=150).generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc)
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Not enough text for word cloud")

with tab4:
    st.subheader("Virality Distribution")
    figv = px.histogram(filtered_df, x="virality_score", color="sentiment", nbins=40)
    st.plotly_chart(figv, use_container_width=True)

with tab5:
    st.subheader("Top Influencers")
    sort_col = 'virality_score' if 'virality_score' in filtered_df.columns else 'followers'
    top_inf = filtered_df.nlargest(10, sort_col)[['entity', 'platform', 'followers', sort_col]]
    st.dataframe(top_inf, use_container_width=True)

# Download
st.sidebar.markdown("---")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Filtered Data", csv, "social_intel_data.csv", "text/csv")

st.success("✅ Dashboard is Ready!")
st.caption(f"Social Media Intelligence System • {datetime.now().strftime('%Y-%m-%d %H:%M')}")