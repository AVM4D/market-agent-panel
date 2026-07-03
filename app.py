import asyncio
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import pydantic

load_dotenv()

from agents import Agent, Runner, function_tool

MODEL_NAME = "gemini/gemini-1.5-pro"

if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing Gemini API Key inside your .env file! Please add it to proceed")
    st.stop()


sentimentalist_agent=Agent(
    name="Sentimentalist",
    instructions="""
    You are an expert Wall Street financial news analyst and NLP specialist.
    Your sole responsibility is to evaluate market news headlines or text snippets.
    
    1. Filter out emotional chatter and focus purely on macroeconomic impact or earnings facts.
    2. Assess whether the incoming data string is overall Bullish, Bearish, or Neutral.
    3. Keep your reasoning brief and focus on structural market factors.
    """,
    model=MODEL_NAME
)

@function_tool
def calculate_momentum(prices_csv: str) -> str:
    """
    Calculates technical momentum markers from a comma-separated string of historical closing prices.
    Use this tool whenever you need to find numerical trend directions for an asset. 
    """
    try:
        price_list=[float(p.strip()) for p in prices_csv.split(",") if p.strip()]
        df = pd.DataFrame(price_list, columns=["Close"])

        if len(df)>=3:
            velocity=df["Close"].iloc[-1] - df["Close"].iloc[-3]
            direction = "UPWARD" if velocity > 0 else "DOWNWARD"
            return f"Calculated Trend: {direction} (Net Change over 3 intervals: {velocity:.2f})"
        else:
            return "Error: Insufficient data history provided to compute a trend velocity"
    except Exception as e:
        return f"Calculation execution failed due to error: {str(e)}"
    
quant_agent = Agent(
    name="Quantative Analyst",
    instructions="""
    You are a high-frequency algorithmic quantitative trading analyst.
    Your sole responsibility is to track technical indicators and asset trends.
    
    1. You never guess asset trends. You must always run numerical calculations.
    2. Use the 'calculate_momentum' tool to extract trend velocities from incoming price data streams.
    3. State your final output clearly based entirely on the mathematical tool results.
    """,
    tools=[calculate_momentum],
    model=MODEL_NAME
)

class RiskAssessmentReport(pydantic.BaseModel):
    market_sentiment: str = pydantic.Field(description="Must be strictly: BULLISH, BEARISH or NEUTRAL.")
    technical_trend: str = pydantic.Field(description="The asset movement velocity calculated by the quant.")
    risk_level_score: int = pydantic.Field(description="A safety risk rating from 1 (Lowest Risk) to 5 (Extreme Risk).")
    advisor_summary: str = pydantic.Field(description="A concise executive summary advising the trader on next steps.")

risk_officer_agent = Agent(
    name="Risk Officer",
    instructions="""
    You are the Chief Risk Officer and Managing Director of an elite digital hedge fund.
    Your responsibility is to synthesize the findings from both the Sentimentalist and the Quantitative Analyst.
    
    1. Cross-examine the text sentiment analysis against the mathematical asset trends.
    2. Assess hidden structural market vulnerabilities or contradictions (e.g., price going up but news is bad).
    3. Generate a strict risk rating score between 1 and 5.
    4. Compile your final answers exclusively into the required output schema.
    """,
    output_schema=RiskAssessmentReport,
    model=MODEL_NAME
)

def handoff_to_sentimentalist() -> Agent:
    """Transfer control to the Sentimentalist Agent for reading news and analyzing text sentiment."""
    return sentimentalist_agent

def handoff_to_quant() -> Agent:
    """Transfer control to the Quantative Analyst Agent for executing math trend calculations."""
    return quant_agent

def handoff_to_risk_officer() -> Agent:
    """Transfer control to the Chief Risk Officer to compile the final validated structured report."""
    return risk_officer_agent

triage_agent = Agent(
    name="Triage Router",
    instructions="""
    You are the central traffic controller and orchestrator for the financial panel.
    Your sole task is to route incoming data streams to the appropriate specialist.
    
    1. If the user provides raw news text or headlines, hand off to the Sentimentalist.
    2. If the user provides historical price matrices or digits, hand off to the Quantitative Analyst.
    3. Once the specialists have added their findings to the context, route to the Risk Officer for final report generation.
    """,
    tools=[handoff_to_sentimentalist,handoff_to_quant,handoff_to_risk_officer],
    model=MODEL_NAME
)