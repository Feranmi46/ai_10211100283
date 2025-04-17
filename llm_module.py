import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.document_loaders import PDFPlumberLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMHandler:
    def __init__(self):
        self.model_name = "gpt2"
        self.tokenizer = None
        self.model = None
        self.vector_store = None
        self.qa_chain = None
        
    def initialize_model(self):
        """Initialize the LLM model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="auto"
            )
            
            # Create a text generation pipeline with adjusted parameters
            text_generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,  # Increased token limit for longer responses
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id  # Ensure proper padding
            )
            
            # Wrap the pipeline in a LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
            return True
        except Exception as e:
            st.error(f"Error initializing model: {str(e)}")
            return False
    
    def load_documents(self, dataset_path):
        """Load and process documents for RAG"""
        try:
            # Check if file exists
            if not os.path.exists(dataset_path):
                # Try to find the file with similar name
                similar_files = [f for f in os.listdir('.') if f.lower().startswith('2025-budget') and f.lower().endswith('.pdf')]
                if similar_files:
                    st.warning(f"File not found: {dataset_path}. Did you mean: {similar_files[0]}?")
                    dataset_path = similar_files[0]
                else:
                    st.error(f"File not found: {dataset_path}. Please check the file path and try again.")
                    return False
                
            # Load documents based on file extension
            if dataset_path.endswith('.pdf'):
                loader = PDFPlumberLoader(dataset_path)
            elif dataset_path.endswith('.csv'):
                loader = CSVLoader(dataset_path)
            else:
                st.error(f"Unsupported file format: {os.path.splitext(dataset_path)[1]}. Please use .pdf or .csv files.")
                return False
                
            documents = loader.load()
            
            if not documents:
                st.error("No content could be loaded from the file. The file might be empty or corrupted.")
                return False
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store using a different model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Using a smaller, more widely available model
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            self.vector_store = FAISS.from_documents(chunks, embeddings)
            
            return True
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")
            return False
    
    def setup_qa_chain(self):
        """Set up the QA chain with RAG"""
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,  # Use the wrapped LLM
                chain_type="stuff",
                retriever=self.vector_store.as_retriever()
            )
            return True
        except Exception as e:
            st.error(f"Error setting up QA chain: {str(e)}")
            return False
    
    def get_answer(self, question):
        """Get answer for a given question"""
        try:
            if not self.qa_chain:
                return "QA chain not initialized. Please initialize the model first."
            
            result = self.qa_chain({"query": question})
            return result["result"]
        except Exception as e:
            return f"Error getting answer: {str(e)}" 