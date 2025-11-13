import streamlit as st
import sqlite3
import random
import datetime
from transformers import pipeline, set_seed
from sentence_transformers import SentenceTransformer, util
import gtts
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import time 

# --- Configuration ---
MIN_WORD_COUNT_BEFORE_INIT = 100   # Target pool size for SAT/GRE words (lowered for faster app startup)
BULK_GENERATE_COUNT = 100        # Number of words to generate in one bulk operation

# ---------- 1. PAGE CONFIG & STYLING ----------
st.set_page_config(
    page_title="SAT/GRE Vocab Master",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved display and professionalism
st.markdown("""
<style>
/* General Background and Containers */
.stApp {
    background-color: #f0f2f6; /* Light gray background */
}

/* Tab styling for better visibility */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
}
.stTabs [data-baseweb="tab-list"] button {
    font-size: 18px;
    font-weight: bold;
    color: #007bff; /* Primary blue color */
    background-color: #fff;
    border-radius: 8px 8px 0 0;
}

/* Card Styling for Vocab */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] > div > [data-testid="stHorizontalBlock"] > div > .stContainer {
    padding: 15px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    transition: all 0.2s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Set global seed for reproducibility of AI generation
set_seed(42)

# --- UTILITIES ---

def format_db_result(result):
    """Converts (word, definition, difficulty) tuple to a dict for better handling."""
    return [{"word": r[0], "definition": r[1], "difficulty": r[2]} for r in result]

def logout():
    """Logs the user out and resets the state."""
    st.session_state.logged_in = False
    st.session_state.login_trigger = False
    st.session_state.clear()
    st.rerun()

# ---------- 2. AI & DB SETUP ----------

@st.cache_resource
def get_db():
    """Initializes and connects to the SQLite database."""
    conn = sqlite3.connect("vocab.db", check_same_thread=False)
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS words(
        word TEXT PRIMARY KEY,
        definition TEXT,
        difficulty INTEGER DEFAULT 0,  -- SRS level: 0 (New) to 5 (Long-term)
        learned INTEGER DEFAULT 0,    -- 1 means fully mastered (difficulty 6+)
        added TEXT,
        next_review_date TEXT
    );
    CREATE TABLE IF NOT EXISTS progress(
        date TEXT UNIQUE,
        learned_count INTEGER DEFAULT 0
    );
    """)
    conn.commit()
    return conn

db = get_db()

# --- DB CRUD FUNCTIONS & SRS LOGIC ---
# (Rest of your file unchanged)