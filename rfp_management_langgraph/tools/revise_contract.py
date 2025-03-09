import os
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from crewai.tools import tool

# ✅ Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)  # Lower temperature for precise legal adjustments

# ✅ Define file paths
DOCUMENTS_DIR = "./outputs/"
CONTRACT_FILE = "6.final_contract.md"
REVIEW_FILE = "7.contract_review.md"

def read_markdown_file(file_path):
    """Reads the contents of a markdown file."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found!")
        return ""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

@tool
def generate_revised_contract():
    """
    CrewAI tool to revise the final contract by **incorporating review feedback** with **minimal structure changes**.
    """

    # ✅ Read input markdown files
    contract_text = read_markdown_file(os.path.join(DOCUMENTS_DIR, CONTRACT_FILE))
    review_feedback = read_markdown_file(os.path.join(DOCUMENTS_DIR, REVIEW_FILE))

    if not contract_text or not review_feedback:
        return "⚠️ Error: Missing contract or review feedback file."

    # ✅ Combine context for LLM
    context = f"""
    **Final Contract (Before Review):**
    {contract_text}

    **Contract Review Feedback:**
    {review_feedback}
    """

    # ✅ Define LLM prompt
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are a **legal contract expert** with deep experience in supplier agreements and procurement law.

        Your task: **Incorporate all review feedback into the final contract** while ensuring the original **format, structure, and intent** remain intact.

        ---
        {context}
        ---

        **Instructions:**
        - **Only modify sections where corrections are needed based on the review.**
        - **DO NOT restructure, rephrase excessively, or introduce unnecessary changes.**
        - **Make concise, legally accurate modifications without altering the original contract format.**
        - **Ensure markdown formatting remains unchanged.**
        
        **Now, apply the review feedback and return the fully revised contract in markdown format.**
        """
    )

    # ✅ Generate revised contract using LLM
    chain = prompt_template | llm
    revised_contract_content = chain.invoke({"context": context})
    revised_contract_content = (
        revised_contract_content.content if hasattr(revised_contract_content, "content") else revised_contract_content
    )

    return revised_contract_content
