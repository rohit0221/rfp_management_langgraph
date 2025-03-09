import os
import fitz  # PyMuPDF for reading PDFs
import chromadb
import re
from dotenv import load_dotenv
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Use the updated package for embeddings
from langchain_core.tools import tool

# Load environment variables (ensure OPENAI_API_KEY and EMBEDDING_MODEL are set in your .env file)
load_dotenv()

# Initialize the OpenAI client (for generating embeddings)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_metadata(text):
    """
    Extracts supplier metadata from text using regex.
    Returns a dictionary with supplier, contact_person, and email.
    """
    metadata = {}
    supplier_match = re.search(r"Company Name:\s*([\w\s]+)", text)
    contact_match = re.search(r"Contact:\s*([\w\s]+)", text)
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    
    metadata["supplier"] = supplier_match.group(1).strip() if supplier_match else "Unknown"
    metadata["contact_person"] = contact_match.group(1).strip() if contact_match else "Unknown"
    metadata["email"] = email_match.group(0) if email_match else "Unknown"
    
    return metadata

@tool
def process_and_store_pdfs(pdf_dir: str):
    """
    Reads PDF files from a directory, extracts full text and metadata,
    splits text into chunks, generates embeddings manually using OpenAIEmbeddings,
    and stores documents, embeddings, and metadata in ChromaDB.
    
    Args:
        pdf_dir (str): The directory containing PDF files.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Create or get a collection without a built-in embedding function (since we're generating embeddings manually)
    collection = chroma_client.get_or_create_collection(name="rfp_proposals")
    
    # Set up the text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    
    # Lists for batch insertion into ChromaDB
    all_ids, all_documents, all_embeddings, all_metadatas = [], [], [], []
    
    # Process each PDF in the directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            doc = fitz.open(file_path)
            text = "\n".join([page.get_text("text") for page in doc])
            
            # Extract metadata from the full document text
            metadata = extract_metadata(text)
            
            # Split the full text into chunks
            chunks = text_splitter.split_text(text)
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_chunk_{i}"
                # Generate embedding using OpenAIEmbeddings (manual embedding)
                embedding = OpenAIEmbeddings(model="text-embedding-ada-002").embed_query(chunk)
                
                if embedding is None:
                    print(f"Embedding generation failed for {chunk_id}. Skipping.")
                    continue
                
                all_ids.append(chunk_id)
                all_documents.append(chunk)
                all_embeddings.append(embedding)
                all_metadatas.append(metadata)
                
                print(f"Processed {chunk_id} | Supplier: {metadata.get('supplier', 'Unknown')}")
    
    # Batch insert into ChromaDB if we have any valid chunks
    if all_ids:
        collection.add(
            ids=all_ids,
            documents=all_documents,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )
        print(f"Successfully stored {len(all_ids)} chunks in ChromaDB with embeddings and metadata.")
        return {"status": "success", "processed_files": len(all_ids)}
    else:
        print("No valid chunks to store.")