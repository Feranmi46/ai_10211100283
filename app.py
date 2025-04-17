import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
from llm_module import LLMHandler

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="AI & ML Explorer",
    page_icon="🤖",
    layout="wide"
)

# Initialize LLM handler
if 'llm_handler' not in st.session_state:
    st.session_state.llm_handler = LLMHandler()

# Sidebar navigation
st.sidebar.title("AI & ML Explorer")
page = st.sidebar.radio(
    "Select Task",
    ["Regression", "Clustering", "Neural Network", "LLM Q&A"]
)

# Regression Page
if page == "Regression":
    st.title("Regression Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Select target column
        target_col = st.selectbox("Select target column", df.columns)
        
        # Select features
        features = st.multiselect("Select features", df.columns.drop(target_col))
        
        if features and target_col:
            # Convert categorical columns to numerical
            X = pd.get_dummies(df[features])
            y = pd.to_numeric(df[target_col], errors='coerce')
            
            # Drop rows with NaN values
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            # Train model
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            
            # Display metrics
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.2f}")
            with col2:
                st.metric("R² Score", f"{r2:.2f}")
            
            # Plot
            fig = px.scatter(x=y, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'})
            st.plotly_chart(fig)
            
            # Custom prediction
            st.subheader("Make Custom Prediction")
            custom_input = {}
            for feature in features:
                if df[feature].dtype == 'object':
                    # For categorical features, show dropdown
                    options = df[feature].unique()
                    custom_input[feature] = st.selectbox(f"Select {feature}", options)
                else:
                    # For numerical features, show number input
                    custom_input[feature] = st.number_input(f"Enter {feature}")
            
            if st.button("Predict"):
                # Convert categorical input to one-hot encoding
                custom_df = pd.DataFrame([custom_input])
                custom_df = pd.get_dummies(custom_df)
                
                # Ensure all columns from training are present
                for col in X.columns:
                    if col not in custom_df.columns:
                        custom_df[col] = 0
                
                # Reorder columns to match training data
                custom_df = custom_df[X.columns]
                
                prediction = model.predict(custom_df)
                st.success(f"Predicted {target_col}: {prediction[0]:.2f}")

# Clustering Page
elif page == "Clustering":
    st.title("Clustering Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Select features
        features = st.multiselect("Select features for clustering", df.columns)
        
        if features:
            X = df[features]
            
            # Number of clusters
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters)
            df['cluster'] = kmeans.fit_predict(X)
            
            # Visualize clusters
            if len(features) >= 2:
                fig = px.scatter(df, x=features[0], y=features[1], color='cluster')
                st.plotly_chart(fig)
            
            # Download clustered data
            st.download_button(
                label="Download clustered data",
                data=df.to_csv(index=False),
                file_name="clustered_data.csv",
                mime="text/csv"
            )

# Neural Network Page
elif page == "Neural Network":
    st.title("Neural Network Training")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())
        
        # Select target column
        target_col = st.selectbox("Select target column", df.columns)
        
        # Select features
        features = st.multiselect("Select features", df.columns.drop(target_col))
        
        if features and target_col:
            # Convert categorical columns to numerical
            X = pd.get_dummies(df[features])
            y = pd.to_numeric(df[target_col], errors='coerce')
            
            # Drop rows with NaN values
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            # Model parameters
            epochs = st.slider("Number of epochs", 1, 100, 10)
            learning_rate = st.slider("Learning rate", 0.001, 0.1, 0.01)
            
            # Build model
            model = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1)
            ])
            
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                         loss='mse',
                         metrics=['mae'])
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train model
            history = model.fit(X, y, epochs=epochs, verbose=0)
            
            # Plot training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            ax1.plot(history.history['loss'])
            ax1.set_title('Loss')
            ax2.plot(history.history['mae'])
            ax2.set_title('MAE')
            st.pyplot(fig)

# LLM Q&A Page
elif page == "LLM Q&A":
    st.title("LLM Question & Answer")
    
    # Dataset selection and handling
    dataset_option = st.selectbox(
        "Select Dataset",
        ["Academic City Student Policy", "Ghana Election Results", "2025 Budget Statement"]
    )
    
    # Initialize LLM handler if not already done
    if 'llm_handler' not in st.session_state:
        st.session_state.llm_handler = LLMHandler()
    
    # Handle dataset selection
    if dataset_option:
        # Initialize LLM
        if st.button("Initialize LLM"):
            with st.spinner("Loading LLM and processing documents..."):
                # Initialize model
                if st.session_state.llm_handler.initialize_model():
                    # Load and process documents based on selection
                    if dataset_option == "Academic City Student Policy":
                        dataset_path = "student_policy.pdf"  # You need to provide this file
                    elif dataset_option == "Ghana Election Results":
                        dataset_path = "Ghana_Election_Result.csv"  # You need to provide this file
                    else:  # 2025 Budget Statement
                        dataset_path = "2025-BudgetStatement-and-Economic-Policy_v4.pdf"  # You need to provide this file
                    
                    if st.session_state.llm_handler.load_documents(dataset_path):
                        if st.session_state.llm_handler.setup_qa_chain():
                            st.success("LLM initialized successfully!")
                        else:
                            st.error("Failed to setup QA chain")
                    else:
                        st.error("Failed to process documents")
                else:
                    st.error("Failed to initialize LLM")
    
    # Question input and answer generation
    if hasattr(st.session_state, 'llm_handler') and st.session_state.llm_handler.qa_chain is not None:
        question = st.text_input("Enter your question:")
        
        if question:
            with st.spinner("Generating answer..."):
                answer = st.session_state.llm_handler.get_answer(question)
                st.write("Answer:", answer)

# Add footer
st.markdown("---")
st.markdown("AI & ML Explorer - Created with Streamlit") 