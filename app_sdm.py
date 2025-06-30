import streamlit as st
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================
# Constants & Paths
# =====================
BASE = r"C:\Users\LENOVO\OneDrive\Desktop\Project_root"
MODEL_DATA_DIR = os.path.join(BASE, "model_data")

# =====================
# Streamlit App
# =====================
# Set page config first
st.set_page_config(layout="wide")
st.title("📍 Predicted Species Distribution Map")

# Check if directory exists
if not os.path.exists(MODEL_DATA_DIR):
    st.error(f"Model data directory not found at: {MODEL_DATA_DIR}")
    st.stop()

# Get species list
try:
    species_list = [f.replace("_model_data.csv", "") for f in os.listdir(MODEL_DATA_DIR) 
                   if f.endswith("_model_data.csv")]
    if not species_list:
        st.error("No species data files found in the directory")
        st.stop()
except Exception as e:
    st.error(f"Error accessing model data directory: {str(e)}")
    st.stop()

species = st.sidebar.selectbox("🔍 Choose Species", species_list)

file_path = os.path.join(MODEL_DATA_DIR, f"{species}_model_data.csv")

# Load data
try:
    df = pd.read_csv(file_path)
    if df.empty:
        st.error(f"The dataset for {species} is empty")
        st.stop()
except Exception as e:
    st.error(f"Error loading data for {species}: {str(e)}")
    st.stop()

# Check required columns
required_columns = {"species", "latitude", "longitude", "presence"}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    st.error(f"Missing required columns in data: {', '.join(missing)}")
    st.stop()

# Prepare features and target
try:
    X = df.drop(columns=["species", "latitude", "longitude", "presence"])
    y = df["presence"]
    
    if X.empty:
        st.error("No features available after dropping columns")
        st.stop()
        
    # Train model and make predictions
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)  # Using all data for final prediction
    df["predicted"] = model.predict(X)
    
    # Filter predicted presence
    predicted_presence = df[df["predicted"] == 1][["latitude", "longitude"]]
    
    # Display results
    st.subheader(f"📍 Predicted Presence Locations for {species}")
    st.map(predicted_presence)

except Exception as e:
    st.error(f"Error during model training/prediction: {str(e)}")
    st.stop()

with st.expander("ℹ️ About the Map"):
    st.markdown("""
    This map shows all locations where the model predicts the species is likely to occur.
    These predictions are made using a trained Random Forest classifier based on environmental variables.
    
    - Each point represents a location where the species is predicted to be present
    - Zoom in/out using mouse wheel or touch gestures
    - Click on points for more details
    """)