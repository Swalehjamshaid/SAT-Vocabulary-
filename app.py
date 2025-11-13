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
MIN_WORD_COUNT_BEFORE_INIT = 2000 # Target pool size for SAT/GRE words
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

def add_word(word, definition):
    """Adds a new word to the database with a default SRS level."""
    today = datetime.date.today().isoformat()
    try:
        # difficulty=0, learned=0, next_review_date=NULL (due immediately)
        db.execute(
            "INSERT OR IGNORE INTO words(word, definition, difficulty, added) VALUES(?, ?, 0, ?)",
            (word, definition, today)
        )
        db.commit()
        st.success(f"Word '{word}' added to your learning list!")
        st.session_state.vocab_page = 0 
    except Exception as e:
        st.error(f"DB Error while adding word: {e}")

def update_word_srs(word, is_correct=True):
    """Updates the word's difficulty and next review date based on the spaced repetition system (SRS)."""
    try:
        # Fetch current difficulty
        current_difficulty = db.execute("SELECT difficulty FROM words WHERE word = ?", (word,)).fetchone()[0]
        
        if is_correct:
            new_difficulty = current_difficulty + 1
            
            if new_difficulty >= 6:
                # Mastered (learned=1)
                db.execute("UPDATE words SET learned = 1, difficulty = ?, next_review_date = NULL WHERE word = ?", (new_difficulty, word))
                
                # Update progress log only on mastery
                today = datetime.date.today().isoformat()
                db.execute(
                    "INSERT INTO progress(date, learned_count) VALUES(?, 1) "
                    "ON CONFLICT(date) DO UPDATE SET learned_count = learned_count + 1",
                    (today,)
                )
                db.commit()
                return "mastered"
            else:
                # Calculate next review date: 1, 3, 7, 15, 30 days
                days_to_add = [0, 1, 3, 7, 15, 30][new_difficulty]
                review_date = (datetime.date.today() + datetime.timedelta(days=days_to_add)).isoformat()
                db.execute("UPDATE words SET difficulty = ?, next_review_date = ? WHERE word = ?", (new_difficulty, review_date, word))
                db.commit()
                return "promoted"
        else:
            # Incorrect answer in quiz: reset difficulty and review today
            db.execute("UPDATE words SET difficulty = 0, next_review_date = ? WHERE word = ?", (datetime.date.today().isoformat(), word))
            db.commit()
            return "demoted"
            
    except Exception as e:
        st.error(f"SRS Update Error: {e}")
        return "error"

def mark_as_learned(word):
    """Marks a word as manually mastered from the Daily Review tab (jumps to mastery level)."""
    try:
        # Set difficulty to 6 (mastered)
        db.execute("UPDATE words SET difficulty = 6, learned = 1, next_review_date = NULL WHERE word = ?", (word,))
        
        today = datetime.date.today().isoformat()
        db.execute(
            "INSERT INTO progress(date, learned_count) VALUES(?, 1) "
            "ON CONFLICT(date) DO UPDATE SET learned_count = learned_count + 1",
            (today,)
        )
        db.commit()
        st.success(f"🥳 Successfully mastered: **{word}**! Reloading list...")
        st.session_state['refresh_vocab_list'] = True
    except Exception as e:
        st.error(f"DB Error while marking word: {e}")


def get_total_unlearned_words():
    """Gets the total count of words pending review (learned=0 and due today)."""
    today = datetime.date.today().isoformat()
    return db.execute(
        f"SELECT COUNT(*) FROM words WHERE learned = 0 AND (next_review_date IS NULL OR next_review_date <= '{today}')"
    ).fetchone()[0]

def get_total_words():
    """Gets the total count of all words in the database (learned and unlearned)."""
    return db.execute("SELECT COUNT(*) FROM words").fetchone()[0]

def get_weekly_words(limit=10, offset=0):
    """Fetches words due for review (or new words) for daily study with offset."""
    today = datetime.date.today().isoformat()
    return db.execute(
        f"SELECT word, definition, difficulty FROM words "
        f"WHERE learned = 0 AND (next_review_date IS NULL OR next_review_date <= '{today}') "
        f"ORDER BY difficulty ASC, next_review_date ASC, added ASC LIMIT {limit} OFFSET {offset}"
    ).fetchall()

def get_stats():
    """Calculates total mastered words and monthly progress."""
    try:
        total = db.execute("SELECT COUNT(*) FROM words WHERE learned = 1").fetchone()[0]
        month = db.execute(
            "SELECT COALESCE(SUM(learned_count), 0) FROM progress "
            "WHERE date >= date('now', '-30 days')"
        ).fetchone()[0]
        
        last_7_days_data = db.execute(
            "SELECT date, learned_count FROM progress "
            "WHERE date >= date('now', '-7 days') ORDER BY date"
        ).fetchall()
        
        date_list = [
            (datetime.date.today() - datetime.timedelta(days=i)).isoformat() 
            for i in range(7)
        ]
        date_list.reverse()
        
        data_map = {date: count for date, count in last_7_days_data}
        chart_data = [data_map.get(d, 0) for d in date_list]
        
        return total, month, date_list, chart_data
    except Exception as e:
        st.error(f"Error fetching stats: {e}")
        return 0, 0, [], []

# --- AI MODEL LOADERS ---

@st.cache_resource
def load_generator():
    """Loads the GPT-2 text generation pipeline."""
    # Note: Setting device to -1 (CPU) for compatibility in most environments
    return pipeline(
        "text-generation",
        model="gpt2",
        max_length=100,
        pad_token_id=50256,
        truncation=True,
        return_full_text=False,
        device=-1 
    )

gen = load_generator()

# --- AI GENERATION FUNCTIONS ---

def generate_sat_word():
    """Generates a high-level vocabulary word (SAT/GRE) and its definition."""
    prompts = [
        "Generate a challenging SAT or GRE vocabulary word: ", 
        "Suggest a highly sophisticated vocabulary word for advanced tests: "
    ]
    prompt = random.choice(prompts)
    try:
        result = gen(prompt, max_length=50, num_return_sequences=1)[0]["generated_text"]
        
        # Basic parsing to isolate the word
        word_candidates = result.strip().split()
        word = next((w.lower() for w in reversed(word_candidates) if w.isalpha() and len(w) > 3), None)
        
        if not word: return None, None
        
        # Add a short delay to prevent transformer overloading during bulk ops
        time.sleep(0.1) 

        def_result = gen(f"Define the SAT/GRE word '{word}' in one clear, concise sentence: ", max_length=70, num_return_sequences=1)[0]["generated_text"]
        
        # Simple cleanup of definition text
        clean_definition = def_result.strip().split('\n')[0].replace(word, 'it').capitalize()
        return word, clean_definition
    except:
        return None, None # Return None on failure

def generate_sentence(word):
    """Generates a contextual sentence for the word."""
    try:
        return gen(f"Use the SAT/GRE word '{word}' in a clear, complex sentence: ", max_length=100, num_return_sequences=1)[0]["generated_text"]
    except:
        return f"The moment was {word} and quickly forgotten."

def generate_mnemonic(word, definition):
    """Generates a memorable mnemonic device."""
    try:
        return gen(f"Create a short, funny mnemonic for '{word}' meaning '{definition[:30]}...': ", max_length=80, num_return_sequences=1)[0]["generated_text"]
    except:
        return f"Think of '{word}' as a fleeting memory."

def speak_word(word):
    """Generates audio bytes for the word pronunciation."""
    try:
        tts = gtts.gTTS(word, lang="en")
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf
    except:
        return None

# --- DATABASE INITIALIZATION FEATURE ---

def initialize_database_with_ai_words(target_count=MIN_WORD_COUNT_BEFORE_INIT, batch_count=BULK_GENERATE_COUNT):
    """Fetches and stores a bulk of new SAT/GRE words into the database."""
    placeholder = st.empty()
    
    current_count = get_total_words()

    if current_count >= target_count:
        return # Target met

    words_needed = target_count - current_count
    words_to_generate_in_batch = min(batch_count, words_needed)

    placeholder.info(f"⏳ **Database Initialization:** Generating the next batch of {words_to_generate_in_batch} SAT/GRE words. Target: {target_count} words.")
    
    words_to_add = []
    
    for i in range(words_to_generate_in_batch):
        word, definition = generate_sat_word()
        if word and definition and len(word) > 3 and " " not in word:
            # difficulty=0, learned=0, next_review_date=NULL (due immediately)
            words_to_add.append((word, definition, 0, 0, datetime.date.today().isoformat()))
            total_progress = (current_count + i + 1) / target_count
            placeholder.progress(total_progress, text=f"Generated {current_count + i + 1}/{target_count} total words...")
        else:
            if i < words_to_generate_in_batch - 5: continue 

    if words_to_add:
        try:
            db.executemany(
                "INSERT OR IGNORE INTO words(word, definition, difficulty, learned, added) VALUES(?, ?, ?, ?, ?)",
                words_to_add
            )
            db.commit()
            
            new_total = get_total_words()
            if new_total >= target_count:
                 placeholder.success(f"🎉 Database fill complete! {new_total} words are ready for study.")
            else:
                 placeholder.success(f"✅ Added {len(words_to_add)} new words. Current total: {new_total}/{target_count}.")
                 
            # Clear the placeholder after a short delay
            time.sleep(2)
            placeholder.empty()
            st.session_state['refresh_vocab_list'] = True # Trigger main app refresh
        except Exception as e:
            placeholder.error(f"Error saving bulk words: {e}")


# Check if database needs initialization before running the main app
if get_total_words() < MIN_WORD_COUNT_BEFORE_INIT:
    # Run initialization to fill the DB up to the target count
    initialize_database_with_ai_words()


# ---------- 3. APPLICATION SECTIONS ----------

def render_sidebar():
    """Renders the statistics and controls in the sidebar."""
    st.sidebar.title("📈 Your Progress")
    total_learned, month_learned, date_list, chart_data = get_stats()

    col_t, col_m = st.sidebar.columns(2)
    with col_t:
        st.metric(label="Total Mastered Words", value=total_learned, delta_color="normal")
    with col_m:
        st.metric(label="In Last 30 Days", value=month_learned, delta=month_learned, delta_color="normal")
    
    # Plot weekly progress
    st.sidebar.subheader("Mastery in Last 7 Days")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(date_list, chart_data, color='#007bff')
    ax.set_xticks(date_list)
    ax.set_xticklabels([d[5:] for d in date_list], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Count")
    ax.tick_params(axis='y', labelsize=8)
    st.sidebar.pyplot(fig)
    
    st.sidebar.markdown(f"**Total Vocab in Database:** {get_total_words()}")

    st.sidebar.markdown("---")
    if st.sidebar.button("Log Out", use_container_width=True):
        logout()

# Navigation helper functions for Paging
def go_to_next_page():
    st.session_state.vocab_page += 1
    st.session_state['refresh_vocab_list'] = True 
    
def go_to_prev_page():
    if st.session_state.vocab_page > 0:
        st.session_state.vocab_page -= 1
        st.session_state['refresh_vocab_list'] = True 

def render_daily_vocab():
    """Renders the daily word review tab with paging for 10 words at a time, prioritized by SRS."""
    st.title("📚 Daily Review Queue (10 Words per Page)")
    st.info("Words shown here are due for review based on the Spaced Repetition System.")
    
    if 'vocab_page' not in st.session_state:
        st.session_state.vocab_page = 0
        
    page_size = 10 
    total_due = get_total_unlearned_words()
    
    # Logic to handle page being out of bounds
    max_page = max(0, (total_due - 1) // page_size)
    if st.session_state.vocab_page > max_page:
        st.session_state.vocab_page = max_page

    offset = st.session_state.vocab_page * page_size
    words = get_weekly_words(limit=page_size, offset=offset)

    if not words and total_due == 0:
        st.balloons()
        st.success("🎉 All words currently due for review have been studied! Check back tomorrow or use the Quiz tab to test your mastery.")
        return
            
    # Paging controls at the top
    st.markdown("---")
    col_prev, col_info, col_next = st.columns([1, 2, 1])
    
    start_index = offset + 1
    end_index = offset + len(words)
    
    col_info.info(f"Showing words **{start_index} - {end_index}** of **{total_due}** words due for review.")
    
    # Previous button
    if st.session_state.vocab_page > 0:
        col_prev.button("⬅️ Previous 10", on_click=go_to_prev_page, use_container_width=True)
    
    # Next button
    if end_index < total_due:
        col_next.button("Next 10 Words ➡️", on_click=go_to_next_page, type="primary", use_container_width=True)
    
    st.markdown("---")

    # Word Cards (3 columns for cleaner display)
    cols = st.columns(3) 
    
    for i, word_data in enumerate(format_db_result(words)):
        with cols[i % 3]:
            # Use a container for a card effect
            with st.container(border=True):
                st.subheader(word_data['word'].capitalize())
                st.caption(f"SRS Level: {word_data['difficulty']} (0=New, 5=Long-term)")
                st.markdown(f"**Definition:** {word_data['definition']}")
                
                # Audio player
                audio_bytes = speak_word(word_data['word'])
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mpeg", loop=False)
                else:
                    st.warning("Could not generate audio.")

                # Mark as Learned button (jumps straight to mastered)
                st.button(
                    "Mark as Mastered (SRS Level 6)", 
                    key=f"learn_{offset + i}", # Unique key with offset
                    on_click=mark_as_learned, 
                    args=(word_data['word'],),
                    use_container_width=True
                )

def render_vocab_generator():
    """Renders the AI-powered new word generator and details tab."""
    st.title("✨ AI Word Generator")
    st.markdown("Generate and explore new SAT/GRE-level words on demand.")

    if 'current_word' not in st.session_state:
        st.session_state.current_word = None
        st.session_state.current_definition = None

    def generate_new_word_session():
        word, definition = generate_sat_word()
        if word and definition:
            st.session_state.current_word = word
            st.session_state.current_definition = definition
            st.session_state.current_sentence = generate_sentence(word)
            st.session_state.current_mnemonic = generate_mnemonic(word, definition)
        else:
            st.error("AI failed to generate a suitable word. Please try again.")
    
    # Initialization or regeneration
    if st.session_state.current_word is None or st.session_state.get('refresh_generator'):
        generate_new_word_session()
        st.session_state.refresh_generator = False


    col_gen, col_add = st.columns([0.7, 0.3])
    
    if st.session_state.current_word:
        with col_gen:
            st.subheader(st.session_state.current_word.capitalize())
            st.markdown(f"**Definition:** {st.session_state.current_definition}")
        
        with col_add:
            # Action buttons
            st.button("🔄 Generate New Word", on_click=generate_new_word_session, use_container_width=True)
            st.button(
                "➕ Add to Learning Queue", 
                on_click=add_word, 
                args=(st.session_state.current_word, st.session_state.current_definition), 
                use_container_width=True, 
                key="add_word_btn"
            )
        
        st.markdown("---")
        
        # Detailed Context
        with st.expander("Explore the Word (Sentence & Mnemonic)", expanded=True):
            st.markdown(f"**Contextual Sentence:**")
            st.info(st.session_state.current_sentence)
            st.markdown(f"**Mnemonic Device:**")
            st.info(st.session_state.current_mnemonic)
            
            # Audio
            audio_bytes = speak_word(st.session_state.current_word)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mpeg", start_time=0)
    else:
        # Fallback if initial generation failed
        st.error("Failed to generate vocabulary. Please click 'Generate New Word' to retry.")
        st.button("🔄 Generate New Word", on_click=generate_new_word_session, use_container_width=True)


def get_quiz_data(num_choices=4):
    """Fetches a word due for review and 3 random distractors for a multiple-choice quiz."""
    today = datetime.date.today().isoformat()
    
    # 1. Prioritize the word due for review (lowest difficulty, earliest review date)
    word_to_test = db.execute(
        f"SELECT word, definition FROM words "
        f"WHERE learned = 0 AND (next_review_date IS NULL OR next_review_date <= '{today}') "
        f"ORDER BY difficulty ASC, next_review_date ASC, added ASC LIMIT 1"
    ).fetchone()

    if not word_to_test:
        return None, "No words currently available for a quiz! Study some first or check back tomorrow."

    correct_word, correct_definition = word_to_test

    # 2. Get distractors (definitions of other words)
    distractors = db.execute(
        "SELECT definition FROM words WHERE word != ? ORDER BY RANDOM() LIMIT ?",
        (correct_word, num_choices - 1)
    ).fetchall()
    
    if len(distractors) < num_choices - 1:
        return None, "Not enough words in the database to create a comprehensive quiz."

    # 3. Prepare the options and shuffle
    options = [correct_definition] + [d[0] for d in distractors]
    random.shuffle(options)
    
    return {
        "word": correct_word,
        "correct_answer": correct_definition,
        "options": options,
        "answer_index": options.index(correct_definition)
    }, None

def render_quiz():
    """Renders the Vocabulary Quiz tab and handles answer submission."""
    st.title("🧠 Vocabulary Quiz (Memory Test)")
    st.markdown("Select the correct definition for the word displayed below. Your answers update the **Spaced Repetition System (SRS)**.")
    
    # State initialization for quiz
    if 'quiz_data' not in st.session_state or st.session_state.quiz_data is None:
        st.session_state.quiz_data, error = get_quiz_data()
        st.session_state.quiz_submitted = False
        st.session_state.quiz_selection = None
        st.session_state.quiz_error = error
        st.session_state.quiz_feedback = ""
    
    if st.session_state.quiz_error:
        st.warning(st.session_state.quiz_error)
        if st.button("Try Another Word", key="retry_quiz"):
             st.session_state.quiz_data = None
             st.rerun()
        return

    quiz = st.session_state.quiz_data
    
    st.markdown("---")
    st.markdown(f"## What is the meaning of: **{quiz['word'].upper()}**?", unsafe_allow_html=True)
    st.markdown("---")
    
    # Radio buttons for choices
    selection = st.radio(
        "Choose the correct definition:",
        quiz['options'],
        key='quiz_options_radio',
        index=None,
        disabled=st.session_state.quiz_submitted
    )
    
    # Store selection to avoid losing it on button press
    st.session_state.quiz_selection = selection
    
    if st.button("Submit Answer", type="primary", use_container_width=True, disabled=st.session_state.quiz_submitted or st.session_state.quiz_selection is None):
        st.session_state.quiz_submitted = True
        
        is_correct = (selection == quiz['correct_answer'])
        
        # Update SRS logic
        srs_result = update_word_srs(quiz['word'], is_correct=is_correct)
        
        if is_correct:
            st.success(f"✅ Correct! '{quiz['word']}' means: **{quiz['correct_answer']}**")
            if srs_result == "mastered":
                 st.balloons()
                 st.session_state.quiz_feedback = f"**Mastered!** This word has been moved to your long-term memory list."
            else:
                 st.session_state.quiz_feedback = f"**Promoted!** You answered correctly. This word will be reviewed in the next SRS cycle."
        else:
            st.error(f"❌ Incorrect. The correct meaning is: **{quiz['correct_answer']}**")
            st.session_state.quiz_feedback = f"**Demoted!** You answered incorrectly. This word's difficulty has been reset and it is due for review immediately."
    
    if st.session_state.quiz_submitted:
        st.info(st.session_state.quiz_feedback)
        if st.button("Next Quiz Word", use_container_width=True):
            # Reset state for next quiz word
            st.session_state.quiz_data = None 
            st.session_state.quiz_submitted = False
            st.rerun()


def main_app():
    """Main application content once logged in."""
    
    # Check if a refresh was triggered by a callback (like marking a word as learned)
    if st.session_state.get('refresh_vocab_list', False):
        st.session_state['refresh_vocab_list'] = False # Reset flag
        st.rerun() # Trigger the rerun now

    render_sidebar()
    
    tab_daily, tab_quiz, tab_generator = st.tabs([
        "📚 Daily Review (10 Words)", 
        "🧠 Vocabulary Quiz", 
        "✨ Word Generator"
    ])

    with tab_daily:
        render_daily_vocab()

    with tab_quiz:
        render_quiz()

    with tab_generator:
        render_vocab_generator()

# ---------- 4. LOGIN & INITIALIZATION CHECK ----------

# Check if the database initialization placeholder is still active
if get_total_words() < MIN_WORD_COUNT_BEFORE_INIT:
    # If the database initialization is running, we stop the rest of the application.
    # The initialization function itself handles the rerun once complete.
    pass

else:
    # Proceed to login or main app
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        
    if "login_trigger" not in st.session_state:
        st.session_state.login_trigger = False

    if not st.session_state.logged_in:
        st.title("🔒 SAT/GRE Vocab Master Access")
        st.markdown("Please enter any username and password to start your vocabulary session.")
        
        with st.container(border=True):
            st.subheader("Access Portal")
            
            with st.form(key="login_form", clear_on_submit=False):
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                
                if submitted:
                    if username and password:
                        st.session_state.logged_in = True
                        st.success("Logged in successfully! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Please enter both username and password.")

    else:
        main_app()