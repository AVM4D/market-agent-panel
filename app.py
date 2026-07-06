import asyncio
import os
import json
import streamlit as st
import pandas as pd
import pydantic
import litellm
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("GEMINI_API_KEY"):
    st.error("Missing Gemini API Key inside your .env file! Please add it to proceed")
    st.stop()

MODEL_NAME = "ollama/qwen2.5:3b"

def calculate_momentum(prices_csv: str) -> str:
    """Calculates technical momentum markers from historical closing prices."""
    try:
        price_list = [float(p.strip()) for p in prices_csv.split(",") if p.strip()]
        df = pd.DataFrame(price_list, columns=["Close"])

        if len(df) >= 3:
            velocity = df["Close"].iloc[-1] - df["Close"].iloc[-3]
            direction = "UPWARD" if velocity > 0 else "DOWNWARD"
            return f"Calculated Trend: {direction} (Net Change over 3 intervals: {velocity:.2f})"
        else:
            return "Error: Insufficient data history provided to compute a trend velocity"
    except Exception as e:
        return f"Calculation execution failed due to error: {str(e)}"

async def run_analysis_pipeline(news_input: str, price_input: str):

        # Check if running on Streamlit Cloud or locally
    IS_CLOUD = os.environ.get("STREAMLIT_RUNTIME_MOCK") is None and os.environ.get("HOSTNAME") is not None

    if IS_CLOUD:
        MODEL_NAME = "gemini/gemini-2.5-flash"
        os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")
    else:
        # local setup
        MODEL_NAME = "ollama/qwen2.5:3b"

    """Executes the multi-agent panel layers sequentially via LiteLLM."""
    
    # 1. Simulate Persona 1: The Sentimentalist
    sent_prompt = f"You are an expert Wall Street news analyst. Analyze this news text and determine if it is strictly BULLISH, BEARISH, or NEUTRAL:\n{news_input}"
    res_news = litellm.completion(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": sent_prompt}]
    )
    sentiment = res_news.choices[0].message.content

    # 2. Simulate Persona 2: The Quantitative Analyst
    math_analysis = calculate_momentum(price_input)

    # 3. Simulate Persona 3: The Risk Officer (JSON Synthesis Output)
    unified_prompt = f"""
    Synthesize these financial findings into a valid structural JSON object layout.
    [SENTIMENT FINDINGS]: {sentiment}
    [QUANT MATHEMATICS]: {math_analysis}
    """
    
    risk_instructions = """You are a Chief Risk Officer. Cross-examine news against technical calculations, calculate a risk rating from 1 to 5, and output a strict JSON object with keys: "market_sentiment", "technical_trend", "risk_level_score", and "advisor_summary". Do not wrap in markdown tags or include conversational text."""
    
    res_risk = litellm.completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": risk_instructions},
            {"role": "user", "content": unified_prompt}
        ],
        response_format={"type": "json_object"}
    )

    # Mock conversation object layout for Streamlit parser matching
    class MockContext:
        def __init__(self, text):
            class MockMessage:
                def __init__(self, txt):
                    self.content = txt
            self.messages = [MockMessage(txt=text)]

    return MockContext(res_risk.choices[0].message.content)

# --- Streamlit Layout Display View Configuration ---
st.set_page_config(page_title="AI Market Intelligence Panel", layout="wide")
st.title("Autonomous Multi-Agent Market Intelligence Panel")
st.subheader("Simulated Institutional Research Group & Risk Analytics")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("News & Fundamentals Input")
    news_area = st.text_area(
        label="Paste recent market articles or headlines:",
        value="Federal Reserve signals potential rate pauses. Consumer retail demand stays resilient but corporate manufacturing yields indicate minor supply bottlenecks.",
        height=150
    )
with col2:
    st.header("Technical Asset Price Matrix")
    price_area = st.text_input(
        label="Enter comma-separated historical closing prices (oldest to newest):",
        value="150.25, 152.10, 151.80, 154.20, 155.60",
    )

analyze_btn = st.button("Execute Autonomous Multi-Agent Analysis Pipeline", use_container_width=True)

if analyze_btn:
    if not news_area.strip() or not price_area.strip():
        st.warning("Please ensure both the news and price fields are not empty before analyzing")
    else:
        with st.spinner("The Multi-Agent Panel is collaborating and analyzing data"):
            try:
                final_context = asyncio.run(run_analysis_pipeline(news_area, price_area))
                st.success("Multi-Agent Analysis Completed Successfully")
                st.markdown("---")
                final_message = final_context.messages[-1]

                if hasattr(final_message, "content") and final_message.content:
                    raw_text = final_message.content.replace("```json", "").replace("```", "").strip()
                    report = json.loads(raw_text)
                    
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.metric("Market Sentiment", str(report.get("market_sentiment")).upper())
                    with m_col2:
                        st.metric("Risk Score Rating", f"{report.get('risk_level_score')} / 5")

                    st.markdown("---")
                    st.subheader("📈 Technical Momentum & Trend Status")
                    st.success(report.get("technical_trend"))

                    st.markdown("---")
                    st.subheader("📋 Executive Advisor Summary Digest")
                    st.info(report.get("advisor_summary"))
                else:
                    st.error("Failed to extract structured data report fields from the final agent response")

            except Exception as system_err:
                st.error(f"An execution crash occurred during the multi-agent pass: {str(system_err)}")