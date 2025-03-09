import csv
import os
import json
from collections import defaultdict
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from crewai.tools import tool

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

@tool
def pricing_risk_analysis_tool():
    """
    Generates a comprehensive pricing risk analysis report based on historical price data.
    """
    pricing_history = load_pricing_history()
    if not pricing_history:
        return "No valid pricing history found. Please check the input CSV."

    risk_analysis = generate_pricing_risk_report(pricing_history)
    return risk_analysis

def load_pricing_history(csv_file="./data/pricing_history/historical_pricing.csv"):
    """
    Loads historical pricing data from a CSV file with columns:
    Supplier, Year, Service, Price ($).
    Returns a nested dictionary formatted as:
    {
      <Service>: {
        "<Year>": {
          <Supplier>: <Price>
        }
      },
      ...
    }
    """
    pricing_data = defaultdict(lambda: defaultdict(dict))
    
    if not os.path.exists(csv_file):
        print(f"⚠️ Warning: {csv_file} not found!")
        return {}
    
    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_cols = ["Supplier", "Year", "Service", "Price ($)"]
        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
        if missing_cols:
            print(f"⚠️ Warning: Missing columns in {csv_file}: {missing_cols}")
            return {}

        for row in reader:
            supplier = row["Supplier"].strip()
            year = row["Year"].strip()
            service = row["Service"].strip()
            
            try:
                price = float(row["Price ($)"].replace(',', ''))
            except ValueError:
                print(f"⚠️ Warning: Non-numeric price in row: {row}")
                continue

            pricing_data[service][year][supplier] = price
    
    return dict(pricing_data)

def generate_pricing_risk_report(pricing_data: dict):
    """
    Uses an LLM to analyze pricing history and generate a risk report.
    """
    context = json.dumps(pricing_data, indent=2)

    response_schemas = [
        ResponseSchema(name="pricing_risk_report", description="A structured markdown report analyzing pricing trends, risks, and strategic recommendations.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an **AI Financial Risk Analyst** specializing in **pricing strategy and risk assessment**.
        Based on the **historical pricing data** provided below, generate a **comprehensive Pricing Risk Analysis Report**.

        **Guidelines:**
        - Identify **price volatility trends** (stable, fluctuating, extreme variations).
        - Detect **market risks** based on price fluctuations.
        - Assess **supplier pricing consistency** (high-risk suppliers with volatile pricing).
        - Predict **future pricing trends** based on historical patterns.
        - Offer **strategic procurement recommendations** (lock in contracts, diversify suppliers, negotiate better rates).
        - Present the analysis in a **detailed markdown report** format.

        ---
        ## 📊 **Pricing Risk Analysis Report**
        ### **1. Historical Price Trends**
        - Identify key **price movements** for each service.
        - Highlight **most volatile and most stable services**.

        ### **2. Market Risks & Opportunities**
        - Assess **inflationary risks**, seasonal patterns, and demand-driven pricing shifts.
        - Identify potential **opportunities for cost savings**.

        ### **3. Supplier Risk Assessment**
        - Evaluate **pricing consistency among suppliers**.
        - Identify **high-risk suppliers** due to frequent price fluctuations.

        ### **4. Future Price Predictions**
        - Forecast **expected price trends** for the next 4 quarters.
        - Provide confidence levels for predictions.

        ### **5. Strategic Recommendations**
        ✅ **Lock in contracts for stable prices?**
        ✅ **Diversify supplier base to reduce dependency?**
        ✅ **Negotiate better rates based on insights?**

        **Pricing Data:**
        ```json
        {context}
        ```

        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt_template | llm | output_parser
    risk_report = chain.invoke({"context": context})
    
    return risk_report["pricing_risk_report"]