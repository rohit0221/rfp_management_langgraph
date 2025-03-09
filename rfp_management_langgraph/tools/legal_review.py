import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from crewai.tools import tool

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

# Define document directory
DOCUMENTS_DIR = "./outputs/"
DOCUMENTS = [
    "1.rfp_comparative_analysis.md",
    "2.pricing_risk_analysis.md",
    "3.negotiation_charter.md",
    "4.negotiation_email.md",
    "5.counter_offer_email.md",
    "5a.counteroffer_strategy.md",
    "6.final_contract.md",  # The final contract to be reviewed
]

def load_documents():
    """Reads and combines all negotiation documents and the final contract."""
    context_data = ""
    contract_text = ""

    for doc in DOCUMENTS:
        file_path = os.path.join(DOCUMENTS_DIR, doc)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                if "final_contract" in doc:
                    contract_text = content  # Identify the final contract separately
                else:
                    context_data += f"\n### {doc}\n" + content + "\n\n"
        else:
            print(f"⚠️ Warning: {doc} not found!")

    if not contract_text:
        raise ValueError("❌ Error: Final contract document not found in ./outputs/!")

    return contract_text, context_data

# Define structured output schema
response_schemas = [
    ResponseSchema(name="key_deviations", description="List of key deviations and discrepancies found."),
    ResponseSchema(name="recommended_corrections", description="Detailed correction recommendations."),
    ResponseSchema(name="final_verdict", description="Overall review conclusion: Acceptable, Minor Fixes, or Major Revisions."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Define prompt for LLM
prompt_template = PromptTemplate(
    input_variables=["contract", "context"],
    template="""
    You are an **AI-powered legal contract reviewer** specializing in procurement and supplier negotiations.
    
    Your task is to review the **final contract** and compare it against the provided **negotiation documents**.
    Highlight any **deviations, missing clauses, risks, or non-compliance issues**.

    ---
    ## **📌 Review Criteria:**
    - **Pricing & Payment Terms:** Ensure alignment with negotiated pricing & risk mitigation strategies.  
    - **Service Level Agreements (SLAs):** Validate penalties, uptime guarantees, and performance requirements.  
    - **Negotiated Benefits:** Ensure agreed-upon **discounts, incentives, pricing flexibility** are included.  
    - **Regulatory Compliance:** Verify **GDPR, SOC2, ISO27001, and other obligations** are present.  
    - **Dispute Resolution:** Assess arbitration clauses for fairness & clarity.  
    - **Exit Clauses & Contract Flexibility:** Confirm renegotiation, termination, and renewal terms exist.  

    ---
    ## **📜 Reference Negotiation Documents:**
    ```markdown
    {context}
    ```

    ---
    ## **📝 Final Contract:**
    ```markdown
    {contract}
    ```

    ---
    {format_instructions}
    """,
    partial_variables={"format_instructions": format_instructions}
)

# Create a chain
chain = prompt_template | llm | output_parser

@tool
def review_contract():
    """Runs the contract review process using LLM."""
    contract_text, context = load_documents()
    review = chain.invoke({"contract": contract_text, "context": context})

    # ✅ Generate Markdown output with correct formatting
    markdown_output = f"""\
# 📄 Contract Review Report

## 🔍 Key Deviations Identified
{format_as_list(review["key_deviations"])}

## 🛠 Recommended Corrections
{format_as_list(review["recommended_corrections"])}

## ✅ Final Verdict
**{review["final_verdict"]}**
    """

    return markdown_output

def format_as_list(text):
    """Ensures that LLM-generated text is properly formatted into markdown bullet points or numbered lists."""
    formatted_text = "\n".join(
        f"- {line.strip()}" if not line.strip().startswith(("*", "-", "•")) else line.strip()
        for line in text.split("\n") if line.strip()
    )
    return formatted_text
