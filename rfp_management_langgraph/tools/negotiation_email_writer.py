import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from crewai.tools import tool

# ✅ Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

def read_markdown_file(file_path):
    """Reads the contents of a markdown file."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found!")
        return ""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

@tool
def generate_negotiation_email():
    """
    CrewAI tool to generate a professional supplier negotiation email based on RFP analysis, pricing risk, and negotiation strategy.
    """
    # ✅ Read input markdown files
    rfp_analysis = read_markdown_file("./outputs/1.rfp_comparative_analysis.md")
    pricing_risk = read_markdown_file("./outputs/2.pricing_risk_analysis.md")
    negotiation_charter = read_markdown_file("./outputs/3.negotiation_charter.md")
    
    # ✅ Combine context for LLM
    context = f"""
    **RFP Comparative Analysis:**
    {rfp_analysis}
    
    **Pricing Risk Analysis:**
    {pricing_risk}
    
    **Negotiation Charter:**
    {negotiation_charter}
    """
    
    # ✅ Define LLM prompt
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are a **Big 4 Consulting Director** crafting a supplier negotiation email.
        Based on the analysis below, write a **highly professional and strategic email** addressing key negotiation points.
        
        ---
        {context}
        ---
        
        **Email Structure:**
        - **Subject:** Strategic Supplier Engagement: Key Negotiation Points & Next Steps
        - **Salutation**
        - **Introduction (Concise, Impactful)**: Reference supplier proposal & evaluation.
        - **Key Findings from Supplier Evaluation**: Pricing competitiveness, risk factors, unique value proposition.
        - **Areas for Negotiation**: Price adjustments, contract flexibility, SLA enhancements, additional value adds.
        - **Call to Action (CTA)**: Request supplier response & meeting scheduling.
        
        Ensure the email is clear, persuasive, and maintains a **formal tone**.
        """
    )
    
    # ✅ Generate email using LLM
    chain = prompt_template | llm
    email_content = chain.invoke({"context": context})
    
    return email_content.content if hasattr(email_content, "content") else email_content
