import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from crewai.tools import tool

# Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

# Define the file paths in ./outputs/
DOCUMENTS_DIR = "./outputs/"
DOCUMENTS = [
    "1.rfp_comparative_analysis.md",
    "2.pricing_risk_analysis.md",
    "3.negotiation_charter.md",
    "4.negotiation_email.md",
    "5.counter_offer_email.md",
    "5a.counteroffer_strategy.md",
]

def load_documents():
    """Reads and combines negotiation-related documents."""
    context_data = ""
    for doc in DOCUMENTS:
        file_path = os.path.join(DOCUMENTS_DIR, doc)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                context_data += f"\n### {doc}\n" + f.read() + "\n\n"
        else:
            print(f"⚠️ Warning: {doc} not found!")
    return context_data

# Define structured output schema
response_schemas = [
    ResponseSchema(name="contract", description="The final structured contract in markdown format."),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

# Define prompt for LLM
prompt_template = PromptTemplate(
    input_variables=["context"],
    template="""
    You are an AI-powered legal assistant specializing in contract drafting.
    
    Based on the following negotiation data, generate a **comprehensive Master Service Agreement (MSA)**
    between the client and the supplier.
    
    Ensure the contract includes:
    - **Scope of Agreement**
    - **Pricing & Payment Terms**
    - **Service Level Agreements (SLAs)**
    - **Contract Term & Exit Clauses**
    - **Compliance & Data Protection**
    - **Negotiated Benefits**
    - **Dispute Resolution & Governing Law**
    - **Signatures & Acceptance**
    
    Use the structured contract template below as an example:
    
    ```markdown
    # Master Service Agreement (MSA)
    **Between:**  
    __[Client Organization]__ ("Client")  
    __[Supplier Name]__ ("Supplier")
    
    **Effective Date:** [Date]  
    **Contract Term:** [36 months with renegotiation at Month 12]
    
    ## 1. Scope of Agreement
    - Supplier agrees to provide the following services:
      - **Cloud Storage, Compute Instances, AI-Powered Analytics, Security & Compliance, Hybrid Cloud Management**
    - The services shall be delivered per the specifications agreed in the RFP Comparative Analysis.
    - Any modifications must be agreed upon in writing and subject to renegotiation.
    
    ## 2. Pricing & Payment Terms
    - **Base Pricing:**  
      - Cloud Storage: $XX per TB/month  
      - Compute Instances: $XX per vCPU/hour  
      - AI-Powered Analytics: $XX per model execution  
      - Security & Compliance: $XX per policy package  
      - Hybrid Cloud Management: $XX per instance/month  
    - **Volume Discounts:**  
      - 10% discount on Cloud Storage if usage exceeds X TB/month  
      - Tiered pricing model for Compute Instances based on usage
    - **Price Protection Clause:**  
      - Prices will remain fixed for the first **12 months**.  
      - Any price increase beyond **5% per annum** must be negotiated.
    - **Payment Terms:**  
      - Net 30 days from invoice date  
      - Late payment penalty: 1.5% per month  
    
    ## 3. Service Level Agreements (SLAs)
    | Service                  | Uptime Guarantee | Response Time | Resolution Time | Penalty for Breach |
    |--------------------------|----------------|---------------|----------------|---------------------|
    | Cloud Storage           | 99.99%          | 10 minutes    | 2 hours        | 10% service credit  |
    | Compute Instances       | 99.95%          | 15 minutes    | 4 hours        | 5% service credit   |
    | AI-Powered Analytics    | 99.9%           | 30 minutes    | 6 hours        | 5% service credit   |
    | Security & Compliance   | 99.99%          | 15 minutes    | 3 hours        | 10% service credit  |
    
    - If SLAs are breached **three consecutive times**, Client has the right to **terminate the contract without penalty**.
    
    ## 4. Contract Term & Exit Clause
    - **Initial Term:** 36 months  
    - **Early Termination Clause:**
      - Either party may terminate with **6 months’ notice**.
      - Termination without cause requires a **buyout fee** equivalent to 3 months of service charges.
    - **Renewal Terms:** Automatic renewal for 12 months unless either party gives 60 days' notice.
    
    ## 5. Compliance & Data Protection
    - **Regulatory Compliance:** Supplier must adhere to **GDPR, SOC2, ISO 27001, HIPAA** (as applicable).
    - **Data Security:**  
      - Supplier shall provide **encryption (AES-256)** and **regular security audits**.  
      - Supplier must notify Client of any security breaches within **24 hours**.
    - **Intellectual Property Rights:**  
      - Any custom developments made for Client remain **Client’s intellectual property**.
    
    ## 6. Negotiated Benefits
    - Waived fees, early renewal incentives, feature enhancements.
    - Supplier must match or beat competitor pricing.
    
    ## 7. Dispute Resolution & Governing Law
    - Any disputes shall first be resolved through **executive-level mediation**.
    - If unresolved, disputes will be settled via **arbitration** under the **ICC (International Chamber of Commerce)**.
    - This agreement shall be governed by **[State/Country Law]**.
    
    ## 8. Acceptance & Signatures
    **Client Organization**  
    **Signature:** ___________________  
    **Name:** ___________________  
    **Title:** ___________________  
    **Date:** ___________________
    
    **Supplier Name**  
    **Signature:** ___________________  
    **Name:** ___________________  
    **Title:** ___________________  
    **Date:** ___________________
    ```
    
    Now, generate a **fully detailed contract** using the above template while integrating the information provided below:
    
    **Negotiation Data:**
    ```json
    {context}
    ```

    {format_instructions}
    """,
    partial_variables={"format_instructions": format_instructions}
)

# Create a chain
chain = prompt_template | llm | output_parser

@tool
def generate_contract():
    """Generates a structured contract using LLM and negotiation data."""
    context = load_documents()
    contract = chain.invoke({"context": context})
    return contract["contract"]
