import os
from dotenv import load_dotenv
import streamlit as st
from chromadb.config import Settings

load_dotenv()

# Define the folder for storing database
PERSIST_DIRECTORY = st.secrets['PERSIST_DIRECTORY']

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)
