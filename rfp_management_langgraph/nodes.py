from typing import Dict
from tools.pdf_vectorizer import process_and_store_pdfs  # Importing the tool

def proposal_processor(state: Dict) -> Dict:
    """
    Node to process supplier proposal PDFs and store them in ChromaDB.
    """
    pdf_dir = state["input_files"].get("proposal_pdfs", "./data/proposals/")

    # Invoke the tool and check for errors
    result = process_and_store_pdfs.invoke(pdf_dir)

    if result["status"] == "success":
        state["steps"]["ProposalProcessor"] = "completed"
    else:
        state["steps"]["ProposalProcessor"] = "failed"

    return state
