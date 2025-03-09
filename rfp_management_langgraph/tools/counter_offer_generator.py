import os
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from crewai.tools import tool
from rfp_management_crew.utils.output_utils import save_markdown

# ✅ Initialize LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

def read_markdown_file(file_path):
    """Reads the contents of a markdown file."""
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: {file_path} not found!")
        return ""
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def generate_counteroffers():
    """
    CrewAI tool to generate strategic counteroffers based on RFP analysis, pricing risk, negotiation strategy, and negotiation email.
    """
    # ✅ Read input markdown files
    rfp_analysis = read_markdown_file("./outputs/1.rfp_comparative_analysis.md")
    pricing_risk = read_markdown_file("./outputs/2.pricing_risk_analysis.md")
    negotiation_charter = read_markdown_file("./outputs/3.negotiation_charter.md")
    negotiation_email = read_markdown_file("./outputs/4.negotiation_email.md")
    
    # ✅ Combine context for LLM
    context = f"""
    **RFP Comparative Analysis:**
    {rfp_analysis}
    
    **Pricing Risk Analysis:**
    {pricing_risk}
    
    **Negotiation Charter:**
    {negotiation_charter}
    
    **Negotiation Email Sent:**
    {negotiation_email}
    """
    
    # ✅ Define LLM prompt
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are a **Big 4 Consulting Director** responsible for **supplier negotiations**.
        Based on the provided RFP analysis, pricing risks, negotiation charter, and negotiation email, generate **highly strategic counteroffers** to anticipated supplier responses.
        
        ---
        {context}
        ---
        
        **Counteroffer Strategy:**
        - Identify key negotiation areas (pricing, contract flexibility, SLAs, support, implementation fees).
        - Predict possible supplier responses based on industry negotiation trends.
        - Generate **structured counteroffers** that maximize value for our organization.
        
        **Structured Output Format:**
        
        ## 🔹 Strategic Counteroffers
        | **Expected Supplier Response** | **Counteroffer Strategy** |
        |--------------------------------|----------------------------|
        | "Our pricing is fixed" | Offer a longer contract in exchange for a discount. |
        | "We can't change SLAs" | Demand higher uptime with penalty clauses. |
        | "Implementation costs are mandatory" | Request free implementation for multi-year deals. |
        | "No flexibility in contract lock-in" | Push for shorter commitment periods with extensions. |
        
        Ensure that the counteroffers are aligned with **business objectives** and are **persuasive yet realistic**.
        """
    )
    
    # ✅ Generate counteroffers using LLM
    chain = prompt_template | llm
    counteroffer_content = chain.invoke({"context": context})
    counteroffer_content = counteroffer_content.content if hasattr(counteroffer_content, "content") else counteroffer_content
    save_markdown(counteroffer_content, filename="5a.counteroffer_strategy.md")
    print("I am here")
    print("Saving Counter Offer strategy")
    return counteroffer_content

@tool
def generate_final_negotiation_email():
    """
    CrewAI tool to generate the final supplier negotiation email incorporating counteroffers.
    """
    # ✅ Read input markdown files
    counteroffers = generate_counteroffers()
    negotiation_email = read_markdown_file("./outputs/negotiation_email.md")
    
    # ✅ Combine context for LLM
    context = f"""
    **Initial Negotiation Email:**
    {negotiation_email}
    
    **Strategic Counteroffers:**
    {counteroffers}
    """
    
    # ✅ Define LLM prompt
    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are a **Big 4 Consulting Director** responsible for supplier negotiations.
        Based on the initial negotiation email and the strategic counteroffers, generate a **final supplier negotiation email** that:
        
        - Acknowledges previous discussions and supplier response.
        - Presents refined counteroffers in a persuasive and strategic manner.
        - Maintains a **formal and professional tone**.
        - Includes a **strong call to action** for finalizing terms.
        
        **Email Structure:**
        - **Subject:** Refining Our Supplier Engagement – Final Negotiation Terms
        - **Salutation**
        - **Introduction**: Reference previous discussions and supplier engagement.
        - **Revised Offer Details**: Present counteroffers with justification.
        - **Final Call to Action**: Push for agreement or final negotiation round.
        
        Ensure the email is **concise, data-backed, and business-oriented**.
        
        ---
        {context}
        ---
        """
    )
    
    # ✅ Generate final email using LLM
    chain = prompt_template | llm
    final_email_content = chain.invoke({"context": context})
    
    return final_email_content.content if hasattr(final_email_content, "content") else final_email_content
