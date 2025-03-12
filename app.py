# ! pip install gradio langchain pypdf huggingface_hub sentence-transformers

import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import os

# Load environment variables (for Hugging Face API key)
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
hf_api_key = os.getenv("HF_API_KEY")

# Function to process the uploaded PDF and create a QA chain
def process_pdf(file):
    # Load the PDF
    loader = PyPDFLoader(file.name)
    pages = loader.load()

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Split text into chunks of 1000 characters
        chunk_overlap=200,  # Add overlap to avoid losing context
    )
    chunks = text_splitter.split_documents(pages)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = DocArrayInMemorySearch.from_documents(chunks, embeddings)

    # Set up the retrieval-based QA chain
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Use a supported Hugging Face model
        huggingfacehub_api_token=hf_api_key,  # Replace with your Hugging Face API key
        model_kwargs={"temperature": 0, "max_length": 512}  # Optional: Adjust model parameters
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Use "stuff" for small documents
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 5 chunks
    )

    return qa_chain

# Function to answer questions using the QA chain
def answer_question(file, question):
    # Process the PDF and create the QA chain
    qa_chain = process_pdf(file)

    # Query the PDF
    response = qa_chain.run(question)
    return response

# Gradio interface
def gradio_interface(file, question):
    return answer_question(file, question)

# Create the Gradio app
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload PDF"),  # File upload input
        gr.Textbox(label="Ask a question"),  # Text input for the question
    ],
    outputs=gr.Textbox(label="Answer"),  # Text output for the answer
    title="PDF Question Answering System",
    description="Upload a PDF and ask questions about its content.",
)

# Launch the app
interface.launch(share=True)