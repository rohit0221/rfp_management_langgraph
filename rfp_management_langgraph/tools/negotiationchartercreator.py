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
def negotiation_charter_creator_tool():
    """
    Generates a strategic negotiation charter based on supply-demand forecasts
    and AI-driven price predictions.
    """
    historical_data = load_supply_demand_forecast()
    if not historical_data:
        return "No valid historical data found. Please check the input CSV."

    price_forecast = generate_price_forecast_langchain(historical_data)
    negotiation_charter = generate_negotiation_charter(price_forecast)

    return negotiation_charter

def load_supply_demand_forecast(csv_file="./data/demand_data/supply_demand.csv"):
    """
    Loads supply-demand data from a CSV file with columns: Year, Quarter, Service, Demand, Supply.
    Returns a nested dictionary formatted as:
    {
      <Service>: {
        "<Year>-<Quarter>": {
          "Demand": <int/float>,
          "Supply": <int/float>
        }
      },
      ...
    }
    """
    forecast_data = defaultdict(dict)

    if not os.path.exists(csv_file):
        print(f"⚠️ Warning: {csv_file} not found!")
        return {}

    with open(csv_file, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_cols = ["Year", "Quarter", "Service", "Demand", "Supply"]
        missing_cols = [col for col in required_cols if col not in reader.fieldnames]
        if missing_cols:
            print(f"⚠️ Warning: Missing columns in {csv_file}: {missing_cols}")
            return {}

        for row in reader:
            year = row["Year"]
            quarter = row["Quarter"]
            service = row["Service"]

            try:
                demand = float(row["Demand"])
                supply = float(row["Supply"])
            except ValueError:
                print(f"⚠️ Warning: Non-numeric Demand/Supply in row: {row}")
                continue

            year_quarter_key = f"{year}-{quarter}"
            forecast_data[service][year_quarter_key] = {"Demand": demand, "Supply": supply}

    return dict(forecast_data)

def generate_price_forecast_langchain(historical_data: dict):
    """
    Uses an LLM to analyze historical supply-demand trends and predict price changes.
    """
    context = json.dumps(historical_data, indent=2)

    response_schemas = [
        ResponseSchema(name="forecast", description="Dictionary containing service names as keys and forecasted price changes as values.")
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an AI expert in **economic forecasting**.
        Given the **historical supply-demand data** below, analyze trends and predict **price percentage changes** 
        for the next four quarters.

        **Guidelines:**
        - If **demand exceeds supply**, prices **increase**.
        - If **supply exceeds demand**, prices **decrease**.
        - Consider **seasonality and demand fluctuations** over time.
        - Ensure each service has a **predicted percentage change** for the next 4 quarters.

        **Historical Data:**
        ```json
        {context}
        ```

        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt_template | llm | output_parser
    forecast_data = chain.invoke({"context": context})

    return forecast_data

def generate_negotiation_charter(price_forecast: dict):
    """
    Uses an LLM to create a detailed Negotiation Charter based on AI-generated price forecasts.
    """
    context = json.dumps(price_forecast, indent=2)

    response_schemas = [
        ResponseSchema(name="negotiation_charter", description="A structured markdown document with price forecasts, risk analysis, and negotiation strategy."),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = PromptTemplate(
        input_variables=["context"],
        template="""
        You are an **AI Procurement Strategy Expert**. 
        Based on the **forecasted price changes** provided below, generate a **detailed Negotiation Charter** 
        to help the procurement team **strategically negotiate better pricing, contracts, and supplier agreements**.

        **Guidelines:**
        - Analyze the **forecasted price changes** for each service.
        - Identify **high-risk suppliers** (if prices are increasing).
        - Identify **negotiation opportunities** (if prices are dropping).
        - Create a structured markdown report with the following sections:

        ---
        ## 📊 **Negotiation Charter Report**
        ### **1. Price Forecast Analysis**
        - Summarize key **price trends** for each service.
        - Highlight which services **are increasing** in price and which **are decreasing**.

        ### **2. Cost Optimization Strategy**
        - Recommendations for **cost savings** based on forecasted price movements.
        - Should the team **lock in long-term contracts, delay purchases, or renegotiate?** 

        ### **3. Supplier Risk Analysis**
        - Assess **supplier risk levels** based on **historical volatility & forecasted changes**.
        - Identify **which suppliers pose the highest cost risks**.

        ### **4. Supplier Comparison Table**
        | Supplier | Service | Current Price | Forecasted Price Change | Risk Level |
        |----------|---------|--------------|------------------------|------------|
        (Fill this table with insights)

        ### **5. Negotiation Leverage Points**
        - **Identify areas where procurement can push for better deals**.
        - **What incentives can be offered to suppliers?**
        - **Are there bulk discount opportunities?**
        - **Are there alternative suppliers with lower risk?**

        ### **6. AI-Powered Negotiation Recommendations**
        ✅ **List specific negotiation strategies** for each supplier.
        ✅ **Highlight key "asks" and "trade-offs"**.
        ✅ **Summarize the best approach for securing optimal contract terms**.

        **Forecasted Price Changes:**
        ```json
        {context}
        ```

        {format_instructions}
        """,
        partial_variables={"format_instructions": format_instructions}
    )

    chain = prompt_template | llm | output_parser
    negotiation_charter = chain.invoke({"context": context})

    return negotiation_charter["negotiation_charter"]

