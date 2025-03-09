import os
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from crewai.tools import tool

# Load environment variables (ensure OPENAI_API_KEY is set)
load_dotenv()

# ✅ Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="rfp_proposals")

# ✅ Initialize LLM and Embeddings
embedding_function = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

def get_unique_suppliers():
    """
    Retrieves all unique supplier names from metadata stored in ChromaDB.
    """
    results = collection.get(include=["metadatas"])
    suppliers = {metadata["supplier"] for metadata in results["metadatas"] if "supplier" in metadata}
    return list(suppliers)

def retrieve_chunks_for_supplier(supplier_name):
    """
    Retrieves all proposal chunks for a given supplier using metadata filtering in ChromaDB.
    """
    results = collection.get(
        where={"supplier": supplier_name}, 
        include=["documents"]
    )
    return results["documents"] if results and "documents" in results else []

@tool
def supplier_analysis_tool():
    """
    CrewAI tool for retrieving and analyzing supplier proposals, generating a detailed markdown report.
    """
    suppliers = get_unique_suppliers()
    supplier_data = {}
    
    for supplier in suppliers:
        print(f"Processing {supplier}...")
        documents = retrieve_chunks_for_supplier(supplier)
        if not documents:
            print(f"⚠️ No data found for {supplier}, skipping.")
            continue
        extracted_data = extract_supplier_details(supplier, documents)
        supplier_data[supplier] = extracted_data
    
    if not supplier_data:
        return "No valid supplier proposals found. Check vector DB."
    
    report = generate_supplier_comparison_report(supplier_data)
    return report

def extract_supplier_details(supplier_name, documents):
    """
    Uses LLM to extract relevant supplier proposal details in a structured markdown format.
    """
    context = "\n".join(documents)
    
    prompt_template = PromptTemplate(
        input_variables=["supplier", "context"],
        template="""
        Extract structured information from the supplier proposal for {supplier}.
        
        Return the extracted details in the following structured markdown format:
        
        ## 📌 Supplier Profile
        - **Company Name**: 
        - **Headquarters**: 
        - **Contact Person & Email**: 
        - **Website**: 
        - **Years of Experience**: 
        - **Industries Served**: 
        
        ## 🚀 Solution Overview
        - **Solution Name**: 
        - **Main Objective**: 
        - **Key Features**: 
        - **AI/ML Capabilities**: 
        - **Technology Stack**: 
        - **Multi-Cloud Compatibility**: 
        
        ## 📊 Pricing & Licensing
        - **Base Monthly Price**: 
        - **Additional Costs**: 
        - **Enterprise Plan Details**: 
        - **Discounts**: 
        - **Implementation/Setup Fee**: 
        - **Contract Lock-in Period**: 
        
        ## 🔍 Key Features & Capabilities
        - **Security & Compliance**: 
        - **Cloud Orchestration**: 
        - **Support & SLAs**: 
        - **Implementation Timeline**: 
        
        ## ⚠️ AI-Powered Risk Assessment
        - **Pricing Risk**: 
        - **Delivery Risk**: 
        - **Contract Risk**: 
        - **Compliance Risk**: 
        - **Overall Score**: 
        
        ## 🔥 Negotiation & Optimization Strategies
        - **How to leverage weaknesses for better pricing & contract terms.**
        - **Opportunities for negotiation and cost savings.**
        
        ## ✅ Final Recommendation
        - **Top Supplier Recommendation**: 
        - **Key Justifications**: 
        - **Next Steps**: 
        
        Proposal Data:
        ```
        {context}
        ```
        """
    )
    
    chain = prompt_template | llm
    extracted_data = chain.invoke({"supplier": supplier_name, "context": context})
    return extracted_data.content if hasattr(extracted_data, "content") else extracted_data

def generate_supplier_comparison_report(supplier_data):
    """
    Uses LLM to generate a comprehensive markdown report comparing supplier proposals.
    """
    comparison_text = "\n\n".join([f"## {supplier}\n{data}" for supplier, data in supplier_data.items()])
    
    prompt_template = PromptTemplate(
        input_variables=["comparison_text"],
        template="""
        Generate a highly detailed supplier proposal evaluation report in a professional markdown format based on the following extracted data:
        
        {comparison_text}
        
        ---
        ## 📊 EXECUTIVE SUMMARY
        - High-level findings on supplier strengths, weaknesses, and rankings.
        - Key insights on cost-effectiveness, security, SLAs.
        - Top-performing supplier recommendations.
        
        ## 🔍 SUPPLIER STRENGTHS & WEAKNESSES
        - Comparative analysis of each supplier's pros and cons.
        
        ## 🔝 OVERALL SUPPLIER RANKING
        | **Supplier** | **Technical Fit (10)** | **Pricing & Cost (10)** | **SLAs & Support (10)** | **Compliance (10)** | **Overall Score** |
        |-------------|------------------|----------------|----------------|--------------|---------------|
        
        ## 🔍 KEY DIFFERENCES BETWEEN SUPPLIERS
        - A structured comparison table showcasing AI capabilities, SLAs, security, pricing models, and contract terms.
        
        ## ⚠️ AI-POWERED RISK ASSESSMENT
        - Pricing Risk, Contract Risk, Compliance Risk, Delivery Risk presented in an easy-to-understand table format.
        
        ## 🔥 NEGOTIATION & OPTIMIZATION STRATEGIES
        - Detailed strategies for cost savings, risk reduction, and maximizing contract flexibility.
        
        ## ✅ FINAL RECOMMENDATION
        - Best supplier selection based on a structured evaluation.
        - Justifications for selection.
        - Next steps for procurement and contract negotiation.
        
        Return the report in a **professionally formatted markdown style** with tables, bullet points, and clear sections.
        """
    )
    
    chain = prompt_template | llm
    report = chain.invoke({"comparison_text": comparison_text})
    return report.content if hasattr(report, "content") else report
