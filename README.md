# AI & ML Explorer

A comprehensive Streamlit-based application for exploring and solving diverse machine learning and AI problems.

## Features

### 1. Regression Analysis
- Upload CSV datasets
- Select target and feature columns
- Train linear regression models
- View performance metrics (MAE, R²)
- Make custom predictions
- Visualize results with interactive plots

### 2. Clustering
- Upload multi-feature datasets
- Select features for clustering
- Interactive cluster number selection
- Visualize clusters in 2D/3D
- Download clustered results

### 3. Neural Network
- Upload classification datasets
- Configure network parameters
- Real-time training visualization
- Model performance metrics
- Custom prediction interface

### 4. LLM Q&A (RAG Implementation)
- Support for multiple document types
- Pre-trained Mistral-7B model
- Retrieval-Augmented Generation
- Real-time question answering
- Document-specific knowledge base

## Deployment Options

### 1. Streamlit Cloud (Recommended)

1. Create an account on [Streamlit Cloud](https://streamlit.io/cloud)
2. Connect your GitHub repository
3. Configure the deployment:
   - Set the main file path to `app.py`
   - Add the following secrets in the Streamlit Cloud dashboard:
     - `HUGGINGFACE_API_TOKEN` (if using private models)

### 2. Heroku Deployment

1. Create a `Procfile`:
```
web: streamlit run app.py --server.port $PORT
```

2. Create a `runtime.txt`:
```
python-3.11.0
```

3. Deploy to Heroku:
```bash
heroku create your-app-name
git push heroku main
```

### 3. Local Deployment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Environment Variables

Create a `.env` file with the following variables:
```
HUGGINGFACE_API_TOKEN=your_token_here
```

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── llm_module.py       # LLM and document processing logic
├── requirements.txt    # Python dependencies
├── .env               # Environment variables
└── README.md          # Project documentation
```

## Usage

1. **Regression Analysis**
   - Upload a CSV file with your dataset
   - Select the target column and features
   - View model performance and visualizations
   - Make custom predictions

2. **Clustering**
   - Upload a dataset with multiple features
   - Select features for clustering
   - Adjust the number of clusters
   - View and download results

3. **Neural Network**
   - Upload a classification dataset
   - Configure network parameters
   - Monitor training progress
   - Make predictions

4. **LLM Q&A**
   - Select a document source
   - Initialize the LLM
   - Ask questions about the document
   - View generated answers

## LLM Architecture

The LLM implementation uses a Retrieval-Augmented Generation (RAG) approach:

1. **Document Processing**
   - Document loading and chunking
   - Text embedding generation
   - Vector store creation

2. **Model Architecture**
   - Mistral-7B-Instruct-v0.1 as base model
   - FAISS for efficient similarity search
   - LangChain for orchestration

3. **RAG Pipeline**
   - Query processing
   - Context retrieval
   - Answer generation

## Methodology

The LLM implementation follows these steps:

1. **Document Preparation**
   - Load and preprocess documents
   - Split into manageable chunks
   - Generate embeddings

2. **Model Setup**
   - Initialize Mistral model
   - Configure RAG pipeline
   - Set up QA chain

3. **Query Processing**
   - Process user questions
   - Retrieve relevant context
   - Generate answers

## Comparison with ChatGPT

The implemented RAG approach offers several advantages over direct ChatGPT usage:

1. **Domain-Specific Knowledge**
   - Better handling of specialized content
   - More accurate answers for specific documents

2. **Transparency**
   - Clear source of information
   - Better traceability of answers

3. **Customization**
   - Fine-tuned for specific use cases
   - Adaptable to different document types

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 