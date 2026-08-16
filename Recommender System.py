import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="Tourism Recommender", layout="wide")

# Load all saved artifacts
@st.cache_resource
def load_artifacts():
    # Load score matrices
    pred_nn = np.load('pred_nn_matrix.npy')
    hybrid = np.load('hybrid_matrix.npy', allow_pickle=True)
    
    # Load mappings
    with open('user_ids.pkl', 'rb') as f:
        user_ids = pickle.load(f)
    with open('item_ids.pkl', 'rb') as f:
        item_ids = pickle.load(f)
    with open('idx_to_item.pkl', 'rb') as f:
        idx_to_item = pickle.load(f)
    with open('user_to_idx.pkl', 'rb') as f:
        user_to_idx = pickle.load(f)
    with open('train_seen.pkl', 'rb') as f:
        train_seen = pickle.load(f)
    # Get the absolute path to the folder containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, 'attraction_metadata.csv')
    
    # Use the absolute path
    attr_meta = pd.read_csv(csv_file_path)
    return (pred_nn, hybrid, user_ids, item_ids,
        idx_to_item, user_to_idx, train_seen, attr_meta)

# Load data
(pred_nn, hybrid, user_ids, item_ids,
 idx_to_item, user_to_idx, train_seen, attr_meta) = load_artifacts()

# Recommendation function
def recommend_for_user(user_id, score_matrix, top_n=5):
    if user_id not in user_to_idx:
        return None, None
    user_idx = user_to_idx[user_id]
    scores = score_matrix[user_idx].copy()
    seen = train_seen.get(user_idx, set())
    # Exclude already seen items
    for i in seen:
        scores[i] = -np.inf
    top_indices = np.argsort(scores)[::-1][:top_n]
    top_scores = scores[top_indices]
    recommendations = [(idx_to_item[i], scores[i]) for i in top_indices]
    return recommendations, seen

# UI
st.title("🎯 Personalized Tourism Recommender")
st.markdown("Enter a tourist ID to get top attraction recommendations.")

tourist_id = st.text_input("Tourist ID", value="605")
if tourist_id:
    try:
        tourist_id_int = int(tourist_id)
    except ValueError:
        st.error("Please enter a valid integer Tourist ID.")
        st.stop()
    
    if tourist_id_int not in user_to_idx:
        st.warning(f"Tourist ID {tourist_id_int} not found. Showing popular attractions (fallback).")
        # Fallback: you can implement a popularity-based list here
        st.info("For demonstration, please try another ID like 605 or 100.")
        st.stop()
    else:
        # Get recommendations using hybrid model (you can add a selector for other models)
        recommendations, seen = recommend_for_user(tourist_id_int, hybrid, top_n=5)
        
        st.subheader("🌟 Top 5 Recommended Attractions")
        cols = st.columns(5)
        for i, (name, score) in enumerate(recommendations):
            with cols[i]:
                st.markdown(f"**{name}**")
                st.caption(f"Score: {score:.3f}")
                # Show extra info from metadata
                meta = attr_meta[attr_meta['attraction_name'] == name].iloc[0] if not attr_meta[attr_meta['attraction_name'] == name].empty else None
                if meta is not None:
                    st.caption(f"Category: {meta['attraction_category']}")
                    st.caption(f"Level: {meta['attraction_level']}")
