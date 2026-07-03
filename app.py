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