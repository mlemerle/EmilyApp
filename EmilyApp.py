"""
Personal Brand Studio - MVP for Marketing Executives
A personal brand command center for executives, creators, and thought leaders.

Features:
- Voice Note Capture with AI transcription and theme detection
- Multi-format Content Generator (LinkedIn, video scripts, newsletters)
- Content Calendar with publishing cadence suggestions
- Brand Gym with skills diagnosis, learning feed, and implementation prompts
"""

import streamlit as st
import sqlite3
import json
import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import os
import openai
from io import BytesIO
import base64

# Configure Streamlit page
st.set_page_config(
    page_title="Personal Brand Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client (optional - will use fallbacks if not available)
try:
    # Try Streamlit secrets first (for cloud deployment)
    if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
        openai.api_key = st.secrets['OPENAI_API_KEY']
        OPENAI_AVAILABLE = True
    # Fallback to environment variable (for local development)
    elif os.getenv('OPENAI_API_KEY'):
        openai.api_key = os.getenv('OPENAI_API_KEY')
        OPENAI_AVAILABLE = True
    else:
        OPENAI_AVAILABLE = False
except:
    OPENAI_AVAILABLE = False

# Database setup
DATABASE_PATH = "personal_brand_studio.db"

@dataclass
class UserProfile:
    """User profile with tone, posting bandwidth, and interests"""
    name: str
    role: str
    company: str
    tone: str  # professional, casual, inspirational, etc.
    posting_frequency: str  # daily, weekly, bi-weekly
    interests: List[str]
    created_at: str = None

@dataclass
class VoiceNote:
    """Voice note with transcription and detected themes"""
    id: int
    transcript: str
    detected_themes: List[str]
    raw_audio_path: str
    created_at: str
    processed: bool = False

@dataclass
class ContentDraft:
    """Generated content draft in various formats"""
    id: int
    voice_note_id: int
    format_type: str  # linkedin, video_script, newsletter
    content: str
    status: str  # draft, approved, scheduled, published
    scheduled_date: str = None
    created_at: str = None

@dataclass
class BrandAnalysis:
    """Brand analysis and recommendations"""
    id: int
    content_balance: Dict[str, int]
    recommendations: List[str]
    learning_suggestions: List[str]
    implementation_prompts: List[str]
    created_at: str

def init_database():
    """Initialize SQLite database with all required tables"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # User profiles table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT,
            company TEXT,
            tone TEXT,
            posting_frequency TEXT,
            interests TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Voice notes table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS voice_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript TEXT NOT NULL,
            detected_themes TEXT,
            raw_audio_path TEXT,
            processed BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Content drafts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_drafts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            voice_note_id INTEGER,
            format_type TEXT NOT NULL,
            content TEXT NOT NULL,
            status TEXT DEFAULT 'draft',
            scheduled_date TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (voice_note_id) REFERENCES voice_notes (id)
        )
    ''')
    
    # Brand analysis table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS brand_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_balance TEXT,
            recommendations TEXT,
            learning_suggestions TEXT,
            implementation_prompts TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def get_user_profile() -> Optional[UserProfile]:
    """Get the current user profile"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM user_profiles ORDER BY created_at DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return UserProfile(
            name=row[1],
            role=row[2],
            company=row[3],
            tone=row[4],
            posting_frequency=row[5],
            interests=json.loads(row[6]) if row[6] else [],
            created_at=row[7]
        )
    return None

def save_user_profile(profile: UserProfile):
    """Save user profile to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO user_profiles (name, role, company, tone, posting_frequency, interests)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (profile.name, profile.role, profile.company, profile.tone, 
          profile.posting_frequency, json.dumps(profile.interests)))
    conn.commit()
    conn.close()

def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio using OpenAI Whisper or fallback"""
    if OPENAI_AVAILABLE:
        try:
            # Handle both bytes and UploadedFile objects
            if hasattr(audio_bytes, 'read'):
                # It's a file-like object (UploadedFile)
                audio_data = audio_bytes.read()
            else:
                # It's already bytes
                audio_data = audio_bytes
            
            # Save audio temporarily
            temp_audio_path = "temp_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(audio_data)
            
            # Use OpenAI Whisper
            with open(temp_audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
                return transcript.text
        except Exception as e:
            st.error(f"Audio transcription failed: {e}")
            return ""
    else:
        # Fallback: Let user manually enter what they said
        st.warning("⚠️ Audio transcription requires OpenAI API key for automatic processing.")
        st.info("💡 **Workaround**: Please listen to your recording and type what you said below:")
        
        # Show the audio player so they can listen and transcribe manually
        if hasattr(audio_bytes, 'read'):
            st.audio(audio_bytes, format="audio/wav")
        else:
            st.audio(audio_bytes, format="audio/wav")
        
        return ""

def detect_themes(text: str) -> List[str]:
    """Detect themes in content using AI or rule-based approach"""
    if OPENAI_AVAILABLE:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing business content themes. Identify the main themes from the following categories: leadership, product, industry insights, personal story, strategy, innovation, team building, customer success, market trends, company culture. Return only a comma-separated list of relevant themes."},
                    {"role": "user", "content": text}
                ],
                max_tokens=100
            )
            themes = response.choices[0].message.content.strip().split(", ")
            return [theme.strip() for theme in themes if theme.strip()]
        except:
            pass
    
    # Fallback rule-based theme detection
    theme_keywords = {
        "leadership": ["lead", "leader", "manage", "decision", "vision", "strategy", "team"],
        "product": ["product", "feature", "development", "launch", "innovation", "design"],
        "industry insights": ["market", "industry", "trend", "analysis", "forecast", "research"],
        "personal story": ["personal", "experience", "journey", "learned", "story", "challenge"],
        "strategy": ["strategy", "plan", "goal", "objective", "roadmap", "direction"],
        "customer success": ["customer", "client", "success", "satisfaction", "feedback", "support"],
        "team building": ["team", "collaboration", "culture", "hiring", "talent", "growth"],
        "innovation": ["innovation", "creative", "new", "breakthrough", "technology", "future"]
    }
    
    detected_themes = []
    text_lower = text.lower()
    
    for theme, keywords in theme_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            detected_themes.append(theme)
    
    return detected_themes[:3]  # Return top 3 themes

def generate_content(transcript: str, format_type: str, user_tone: str) -> str:
    """Generate content in specified format using AI or templates"""
    
    # Emily's specific LinkedIn style prompt
    emily_linkedin_prompt = """You're a GPT designed to write in a warm, conversational tone with a mix of playful corporate humor, polished storytelling, and confident-but-approachable style. Your writing should feel like a senior leader chatting with a colleague over coffee—part mentor, part meme-sharer. You sound friendly and personable, but you're also sharp and narrative-driven. Prioritize short to medium-length sentences, and lean into a casual rhythm with professional polish. Avoid jargon or cliché phrases like "here's the kicker."

Style Guide:
- Write in first-person, as if narrating your own experience
- Blend personal anecdotes, quick insights, and useful advice
- Be casually witty, slightly self-deprecating, and confidently insightful
- Use vivid, concrete language and sensory phrasing when appropriate
- Switch between short, punchy sentences and longer, flowing ones
- Infuse a multichannel brand energy (LinkedIn, events, personal wins)
- Position achievements with humility and humor (e.g. "wear many hats," "accidentally went viral")
- Think of it like storytelling-meets-LinkedIn-meets-slide-deck-afterparty

Tone: Friendly, energetic, and self-aware. Confident but never arrogant. Lightly pokes fun at corporate norms (while clearly understanding them). A mix of real-life reflections, data points, memes, and insights. You're not just here to inform—you're here to connect, inspire, and make the reader smile (or snort-laugh) while learning something useful.

Create a LinkedIn post based on this content. Keep it under 1300 characters and include relevant hashtags."""
    
    if OPENAI_AVAILABLE:
        try:
            format_prompts = {
                "linkedin": emily_linkedin_prompt,
                "video_script": f"Create a 60-90 second video script in a {user_tone} tone based on this content. Include clear talking points and engagement hooks.",
                "newsletter": f"Create a newsletter snippet in a {user_tone} tone based on this content. Make it informative and actionable for business leaders."
            }
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": format_prompts.get(format_type, f'Create engaging business content in a {user_tone} tone.')},
                    {"role": "user", "content": transcript}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except:
            pass
    
    # Enhanced fallback templates with Emily's style for LinkedIn
    templates = {
        "linkedin": f"""Just had one of those lightbulb moments...

{transcript[:200]}...

Anyone else experiencing this? Drop your thoughts below – I'm genuinely curious what your take is! 👇

(And yes, I may have overthought this during my third coffee of the day ☕)

#Leadership #RealTalk #CorporateLife #Growth""",
        
        "video_script": f"""[HOOK] Okay, so this just happened and I had to share...

[MAIN POINT] {transcript[:150]}...

[PERSONAL TOUCH] Classic case of "learn something new every day," right?

[CALL TO ACTION] What's your experience with this? I'd love to hear your stories in the comments!

[DURATION: 60-90 seconds]""",
        
        "newsletter": f"""💡 **This Week's "Aha!" Moment**

So here's what went down this week...

{transcript[:300]}...

**The Real Talk:** Sometimes the best insights come from the most unexpected places. (Who knew?)

**Your Move:** Take a moment to think about where your latest breakthrough came from. I bet it wasn't where you expected.

Stay curious,
Emily ✨
"""
    }
    
    return templates.get(format_type, transcript)

def save_voice_note(transcript: str, themes: List[str], audio_path: str = None) -> int:
    """Save voice note to database and return ID"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO voice_notes (transcript, detected_themes, raw_audio_path)
        VALUES (?, ?, ?)
    ''', (transcript, json.dumps(themes), audio_path or ""))
    voice_note_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return voice_note_id

def save_content_draft(voice_note_id: int, format_type: str, content: str) -> int:
    """Save content draft to database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO content_drafts (voice_note_id, format_type, content)
        VALUES (?, ?, ?)
    ''', (voice_note_id, format_type, content))
    draft_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return draft_id

def get_content_drafts(status: str = None) -> List[Dict]:
    """Get content drafts from database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    if status:
        cursor.execute("SELECT * FROM content_drafts WHERE status = ? ORDER BY created_at DESC", (status,))
    else:
        cursor.execute("SELECT * FROM content_drafts ORDER BY created_at DESC")
    
    rows = cursor.fetchall()
    conn.close()
    
    drafts = []
    for row in rows:
        drafts.append({
            'id': row[0],
            'voice_note_id': row[1],
            'format_type': row[2],
            'content': row[3],
            'status': row[4],
            'scheduled_date': row[5],
            'created_at': row[6]
        })
    
    return drafts

def update_draft_status(draft_id: int, status: str, scheduled_date: str = None):
    """Update draft status in database"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE content_drafts 
        SET status = ?, scheduled_date = ?
        WHERE id = ?
    ''', (status, scheduled_date, draft_id))
    conn.commit()
    conn.close()

def analyze_brand_balance() -> Dict[str, int]:
    """Analyze content balance across themes"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT detected_themes FROM voice_notes WHERE processed = TRUE")
    rows = cursor.fetchall()
    conn.close()
    
    theme_counts = {}
    for row in rows:
        themes = json.loads(row[0]) if row[0] else []
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
    return theme_counts

def generate_brand_recommendations(theme_counts: Dict[str, int]) -> List[str]:
    """Generate brand recommendations based on content balance"""
    recommendations = []
    
    # Ideal theme distribution for executives
    ideal_themes = {
        "leadership": 25,
        "industry insights": 20,
        "personal story": 15,
        "strategy": 15,
        "innovation": 10,
        "team building": 10,
        "customer success": 5
    }
    
    total_content = sum(theme_counts.values()) or 1
    
    for theme, ideal_percentage in ideal_themes.items():
        current_count = theme_counts.get(theme, 0)
        current_percentage = (current_count / total_content) * 100
        
        if current_percentage < ideal_percentage * 0.7:  # If significantly below ideal
            recommendations.append(f"Consider sharing more {theme} content. You're currently at {current_percentage:.1f}% vs ideal {ideal_percentage}%")
    
    if not recommendations:
        recommendations.append("Great job maintaining balanced content! Keep up the diverse mix of themes.")
    
    return recommendations

def get_learning_suggestions(weak_themes: List[str]) -> List[str]:
    """Get learning suggestions for brand gaps"""
    learning_resources = {
        "leadership": [
            "Read 'The First 90 Days' by Michael Watkins",
            "Watch Simon Sinek's TED Talk on 'Start With Why'",
            "Follow leadership insights from Brené Brown"
        ],
        "industry insights": [
            "Subscribe to McKinsey Global Institute reports",
            "Follow Harvard Business Review weekly summaries",
            "Set up Google Alerts for your industry keywords"
        ],
        "personal story": [
            "Read 'The Storytelling Edge' by Shane Snow",
            "Practice the Hero's Journey framework for business stories",
            "Document your weekly 'lessons learned' moments"
        ],
        "strategy": [
            "Read 'Good Strategy Bad Strategy' by Richard Rumelt",
            "Study case studies from your industry leaders",
            "Follow strategy frameworks from BCG insights"
        ],
        "innovation": [
            "Follow innovation labs from top tech companies",
            "Read 'The Innovator's Dilemma' by Clayton Christensen",
            "Join innovation-focused LinkedIn groups"
        ]
    }
    
    suggestions = []
    for theme in weak_themes:
        if theme in learning_resources:
            suggestions.extend(learning_resources[theme])
    
    return suggestions[:5]  # Return top 5 suggestions

def get_implementation_prompts(weak_themes: List[str]) -> List[str]:
    """Get implementation prompts for immediate action"""
    prompts = {
        "leadership": [
            "Record a story about a recent difficult decision you made and what you learned",
            "Share your framework for giving constructive feedback to team members",
            "Describe a time when you had to pivot strategy and how you communicated it"
        ],
        "personal story": [
            "Tell the story of your biggest career failure and what it taught you",
            "Share what motivated you to join your current company or role",
            "Describe a mentor who shaped your leadership style"
        ],
        "industry insights": [
            "Analyze a recent industry report and share your key takeaways",
            "Predict one major trend that will impact your industry in the next 2 years",
            "Compare your industry today vs. 5 years ago - what's changed?"
        ],
        "strategy": [
            "Explain your approach to quarterly planning and goal setting",
            "Share how you evaluate and prioritize competing initiatives",
            "Describe your framework for making data-driven decisions"
        ]
    }
    
    implementation_prompts = []
    for theme in weak_themes:
        if theme in prompts:
            implementation_prompts.extend(prompts[theme])
    
    return implementation_prompts[:3]  # Return top 3 prompts

def display_voice_note_results(transcript: str, themes: List[str], voice_note_id: int):
    """Display voice note processing results"""
    st.subheader("Transcript")
    st.write(transcript)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detected Themes")
        for theme in themes:
            st.badge(theme)
    
    with col2:
        st.subheader("Next Steps")
        st.info("Go to Content Generator to create posts from this content!")
    
    # Store in session state for content generation
    st.session_state.latest_voice_note = {
        'id': voice_note_id,
        'transcript': transcript,
        'themes': themes
    }

def suggest_posting_cadence(user_frequency: str, content_count: int) -> Dict[str, str]:
    """Suggest optimal posting cadence based on user bandwidth and content"""
    frequency_mapping = {
        "daily": {"posts_per_week": 7, "ideal_content_buffer": 14},
        "every_other_day": {"posts_per_week": 3.5, "ideal_content_buffer": 7},
        "weekly": {"posts_per_week": 1, "ideal_content_buffer": 4},
        "bi-weekly": {"posts_per_week": 0.5, "ideal_content_buffer": 2}
    }
    
    freq_info = frequency_mapping.get(user_frequency, frequency_mapping["weekly"])
    weeks_of_content = content_count / freq_info["posts_per_week"]
    
    suggestions = {
        "current_buffer": f"You have {weeks_of_content:.1f} weeks of content ready",
        "recommendation": "",
        "next_creation_date": ""
    }
    
    if weeks_of_content < 1:
        suggestions["recommendation"] = "🔴 Create more content immediately - you're running low!"
        suggestions["next_creation_date"] = "Today"
    elif weeks_of_content < 2:
        suggestions["recommendation"] = "🟡 Consider creating content this week to maintain buffer"
        suggestions["next_creation_date"] = "This week"
    else:
        suggestions["recommendation"] = "🟢 Good content buffer! You can focus on other priorities"
        suggestions["next_creation_date"] = "Next week"
    
    return suggestions

# Streamlit App UI
def main():
    """Main Streamlit application"""
    
    # Initialize database
    init_database()
    
    # Sidebar navigation
    st.sidebar.title("🎯 Personal Brand Studio")
    st.sidebar.markdown("---")
    
    # Get or create user profile
    user_profile = get_user_profile()
    
    if not user_profile:
        st.sidebar.warning("Please set up your profile first!")
        page = "Profile Setup"
    else:
        st.sidebar.success(f"Welcome, {user_profile.name}!")
        page = st.sidebar.selectbox(
            "Navigate to:",
            [
                "Voice Note Capture",
                "Content Generator", 
                "Content Library",
                "Content Calendar",
                "Brand Gym",
                "Profile Setup"
            ]
        )
    
    # Main content area
    if page == "Profile Setup":
        profile_setup_page()
    elif page == "Voice Note Capture":
        voice_capture_page(user_profile)
    elif page == "Content Generator":
        content_generator_page(user_profile)
    elif page == "Content Library":
        content_library_page()
    elif page == "Content Calendar":
        content_calendar_page(user_profile)
    elif page == "Brand Gym":
        brand_gym_page()

def profile_setup_page():
    """User profile setup page"""
    st.title("👤 Profile Setup")
    st.markdown("Set up your personal brand profile to get personalized content suggestions.")
    
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Full Name*", placeholder="John Smith")
            role = st.text_input("Role/Title*", placeholder="CEO, VP Marketing, etc.")
            company = st.text_input("Company", placeholder="Your company name")
            
        with col2:
            tone = st.selectbox(
                "Content Tone*",
                ["Professional", "Conversational", "Inspirational", "Analytical", "Storytelling"]
            )
            posting_frequency = st.selectbox(
                "Posting Frequency*",
                ["daily", "every_other_day", "weekly", "bi-weekly"]
            )
        
        st.markdown("**Areas of Interest/Expertise** (select all that apply)")
        interests_options = [
            "Leadership", "Strategy", "Innovation", "Technology", 
            "Marketing", "Sales", "Product Management", "Operations",
            "Finance", "HR/People", "Customer Success", "Industry Trends"
        ]
        
        interests = st.multiselect("Interests", interests_options)
        
        submit_button = st.form_submit_button("Save Profile")
        
        if submit_button:
            if name and role and tone and posting_frequency:
                profile = UserProfile(
                    name=name,
                    role=role,
                    company=company,
                    tone=tone.lower(),
                    posting_frequency=posting_frequency,
                    interests=interests
                )
                save_user_profile(profile)
                st.success("Profile saved successfully! Navigate to other pages to start creating content.")
                st.rerun()
            else:
                st.error("Please fill in all required fields marked with *")

def voice_capture_page(user_profile: UserProfile):
    """Voice note capture and processing page"""
    st.title("🎤 Voice Note Capture")
    st.markdown("Record your thoughts or upload audio files to generate content ideas.")
    
    # Tab layout for different input methods
    tab1, tab2, tab3 = st.tabs(["🎤 Record Audio", "📝 Text Input", "🎵 Upload Audio"])
    
    with tab1:
        st.subheader("Record Your Voice Note")
        st.markdown("Click the record button below to capture your thoughts directly in the browser.")
        
        # Audio recorder widget
        audio_bytes = st.audio_input("Record your voice note:")
        
        if audio_bytes is not None:
            st.audio(audio_bytes, format="audio/wav")
            
            # Show transcription options
            st.markdown("**Choose your transcription method:**")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🤖 Auto-Transcribe (OpenAI)", type="primary", disabled=not OPENAI_AVAILABLE):
                    with st.spinner("Transcribing audio and analyzing themes..."):
                        transcript = transcribe_audio(audio_bytes)
                        
                        if transcript and transcript.strip():
                            # Detect themes and process
                            themes = detect_themes(transcript)
                            voice_note_id = save_voice_note(transcript, themes, "recorded_audio.wav")
                            
                            # Display results
                            st.success("Audio processed successfully!")
                            display_voice_note_results(transcript, themes, voice_note_id)
                        else:
                            st.error("Auto-transcription failed. Please use manual transcription below.")
                            st.session_state.manual_transcribe_mode = True
            
            with col2:
                if st.button("✍️ Manual Transcription"):
                    st.session_state.manual_transcribe_mode = True
            
            # Manual transcription mode
            if st.session_state.get('manual_transcribe_mode'):
                st.markdown("**Manual Transcription**")
                st.info("🎧 Listen to your recording above and type what you said:")
                
                manual_transcript = st.text_area(
                    "Type your voice note content:",
                    placeholder="Type what you said in the recording...",
                    height=100,
                    key="manual_transcript_input"
                )
                
                if st.button("Process Manual Transcript") and manual_transcript:
                    with st.spinner("Analyzing themes and generating content..."):
                        # Detect themes
                        themes = detect_themes(manual_transcript)
                        voice_note_id = save_voice_note(manual_transcript, themes, "recorded_audio.wav")
                        
                        # Display results
                        st.success("Content processed successfully!")
                        display_voice_note_results(manual_transcript, themes, voice_note_id)
                        
                        # Clear the manual transcription mode
                        st.session_state.manual_transcribe_mode = False
    
    with tab2:
        st.subheader("Enter Your Thoughts")
        text_input = st.text_area(
            "Share your thoughts, insights, or ideas:",
            placeholder="Type your thoughts here... For example: 'I've been thinking about how AI is changing our industry. We implemented a new tool last month and saw 30% efficiency gains...'",
            height=150
        )
        
        if st.button("Process Text", type="primary"):
            if text_input:
                with st.spinner("Analyzing themes and generating content..."):
                    # Detect themes
                    themes = detect_themes(text_input)
                    
                    # Save voice note
                    voice_note_id = save_voice_note(text_input, themes)
                    
                    # Display results
                    st.success("Content processed successfully!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Detected Themes")
                        for theme in themes:
                            st.badge(theme)
                    
                    with col2:
                        st.subheader("Next Steps")
                        st.info("Go to Content Generator to create posts from this content!")
                    
                    # Store in session state for content generation
                    st.session_state.latest_voice_note = {
                        'id': voice_note_id,
                        'transcript': text_input,
                        'themes': themes
                    }
            else:
                st.error("Please enter some text to process.")
    
    with tab3:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg'],
            help="Upload audio recordings of your thoughts, meeting notes, or voice memos"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("Transcribe & Process Audio", type="primary"):
                with st.spinner("Transcribing audio and analyzing themes..."):
                    # Transcribe audio - uploaded_file is already a file-like object
                    transcript = transcribe_audio(uploaded_file)
                    
                    if transcript and transcript.strip():
                        # Detect themes
                        themes = detect_themes(transcript)
                        
                        # Save voice note
                        voice_note_id = save_voice_note(transcript, themes, uploaded_file.name)
                        
                        # Display results
                        st.success("Audio processed successfully!")
                        display_voice_note_results(transcript, themes, voice_note_id)
                    else:
                        st.error("Transcription failed. Please try manual transcription:")
                        manual_transcript = st.text_area(
                            "Listen to the audio above and type what you heard:",
                            placeholder="Type the content of your audio file...",
                            height=100,
                            key="upload_manual_transcript"
                        )
                        
                        if st.button("Process Manual Transcript", key="upload_process_manual") and manual_transcript:
                            themes = detect_themes(manual_transcript)
                            voice_note_id = save_voice_note(manual_transcript, themes, uploaded_file.name)
                            st.success("Content processed successfully!")
                            display_voice_note_results(manual_transcript, themes, voice_note_id)

def content_generator_page(user_profile: UserProfile):
    """Content generation page"""
    st.title("✨ Content Generator")
    st.markdown("Transform your voice notes into polished content for different platforms.")
    
    # Check if there's a recent voice note to work with
    if 'latest_voice_note' in st.session_state:
        voice_note = st.session_state.latest_voice_note
        
        st.success(f"Working with your latest content: \"{voice_note['transcript'][:100]}...\"")
        
        # Content format selection
        st.subheader("Choose Content Formats")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📱 LinkedIn Post", use_container_width=True):
                generate_and_display_content(voice_note, "linkedin", user_profile.tone)
        
        with col2:
            if st.button("🎥 Video Script", use_container_width=True):
                generate_and_display_content(voice_note, "video_script", user_profile.tone)
        
        with col3:
            if st.button("📧 Newsletter Snippet", use_container_width=True):
                generate_and_display_content(voice_note, "newsletter", user_profile.tone)
        
        # Show existing drafts for this voice note
        show_existing_drafts(voice_note['id'])
        
    else:
        st.info("No recent voice notes found. Please go to Voice Note Capture to create content first.")
        
        # Option to select from existing voice notes
        st.subheader("Or Select from Previous Voice Notes")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, transcript, created_at FROM voice_notes ORDER BY created_at DESC LIMIT 10")
        voice_notes = cursor.fetchall()
        conn.close()
        
        if voice_notes:
            for note in voice_notes:
                with st.expander(f"Voice Note from {note[2]} - \"{note[1][:50]}...\""):
                    st.write(note[1])
                    if st.button(f"Use This Note", key=f"use_note_{note[0]}"):
                        # Get themes for this note
                        conn = sqlite3.connect(DATABASE_PATH)
                        cursor = conn.cursor()
                        cursor.execute("SELECT detected_themes FROM voice_notes WHERE id = ?", (note[0],))
                        themes_json = cursor.fetchone()[0]
                        conn.close()
                        
                        st.session_state.latest_voice_note = {
                            'id': note[0],
                            'transcript': note[1],
                            'themes': json.loads(themes_json) if themes_json else []
                        }
                        st.rerun()

def generate_and_display_content(voice_note: Dict, format_type: str, user_tone: str):
    """Generate and display content for a specific format"""
    with st.spinner(f"Generating {format_type} content..."):
        content = generate_content(voice_note['transcript'], format_type, user_tone)
        
        # Save draft to database
        draft_id = save_content_draft(voice_note['id'], format_type, content)
        
        # Display generated content
        st.subheader(f"{format_type.replace('_', ' ').title()} Draft")
        
        # Editable text area for user modifications
        edited_content = st.text_area(
            "Edit content as needed:",
            value=content,
            height=200,
            key=f"edit_{format_type}_{draft_id}"
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Draft", key=f"save_{draft_id}"):
                # Update content if edited
                if edited_content != content:
                    conn = sqlite3.connect(DATABASE_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE content_drafts SET content = ? WHERE id = ?", 
                                 (edited_content, draft_id))
                    conn.commit()
                    conn.close()
                st.success("Draft saved!")
        
        with col2:
            if st.button("✅ Approve", key=f"approve_{draft_id}"):
                update_draft_status(draft_id, "approved")
                st.success("Draft approved!")
        
        with col3:
            if st.button("📅 Schedule", key=f"schedule_{draft_id}"):
                st.session_state[f"scheduling_{draft_id}"] = True
        
        # Scheduling interface
        if st.session_state.get(f"scheduling_{draft_id}"):
            schedule_date = st.date_input("Schedule for:", key=f"date_{draft_id}")
            schedule_time = st.time_input("Time:", key=f"time_{draft_id}")
            
            if st.button("Confirm Schedule", key=f"confirm_{draft_id}"):
                scheduled_datetime = f"{schedule_date} {schedule_time}"
                update_draft_status(draft_id, "scheduled", scheduled_datetime)
                st.success(f"Scheduled for {scheduled_datetime}")
                st.session_state[f"scheduling_{draft_id}"] = False

def show_existing_drafts(voice_note_id: int):
    """Show existing drafts for a voice note"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM content_drafts WHERE voice_note_id = ?", (voice_note_id,))
    drafts = cursor.fetchall()
    conn.close()
    
    if drafts:
        st.subheader("Existing Drafts")
        for draft in drafts:
            with st.expander(f"{draft[2].replace('_', ' ').title()} - {draft[4].title()}"):
                st.write(draft[3])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Approve", key=f"approve_existing_{draft[0]}"):
                        update_draft_status(draft[0], "approved")
                        st.rerun()
                
                with col2:
                    if st.button("📅 Schedule", key=f"schedule_existing_{draft[0]}"):
                        st.session_state[f"scheduling_existing_{draft[0]}"] = True
                
                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{draft[0]}"):
                        conn = sqlite3.connect(DATABASE_PATH)
                        cursor = conn.cursor()
                        cursor.execute("DELETE FROM content_drafts WHERE id = ?", (draft[0],))
                        conn.commit()
                        conn.close()
                        st.rerun()

def content_library_page():
    """Content library and draft management page"""
    st.title("📚 Content Library")
    st.markdown("Manage all your content drafts and published posts.")
    
    # Filter tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 All Drafts", "✅ Approved", "📅 Scheduled", "✨ Published"])
    
    with tab1:
        show_drafts_by_status(None, "All Drafts")
    
    with tab2:
        show_drafts_by_status("approved", "Approved Drafts")
    
    with tab3:
        show_drafts_by_status("scheduled", "Scheduled Posts")
    
    with tab4:
        show_drafts_by_status("published", "Published Posts")

def show_drafts_by_status(status: str, title: str):
    """Show drafts filtered by status"""
    st.subheader(title)
    
    drafts = get_content_drafts(status)
    
    if not drafts:
        st.info(f"No {title.lower()} found.")
        return
    
    for draft in drafts:
        with st.expander(f"{draft['format_type'].replace('_', ' ').title()} - {draft['status'].title()} - {draft['created_at'][:10]}"):
            st.write(draft['content'])
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if draft['status'] != 'approved':
                    if st.button("✅ Approve", key=f"lib_approve_{draft['id']}"):
                        update_draft_status(draft['id'], "approved")
                        st.rerun()
            
            with col2:
                if draft['status'] not in ['scheduled', 'published']:
                    if st.button("📅 Schedule", key=f"lib_schedule_{draft['id']}"):
                        st.session_state[f"lib_scheduling_{draft['id']}"] = True
            
            with col3:
                if draft['status'] == 'scheduled':
                    if st.button("✨ Mark Published", key=f"lib_publish_{draft['id']}"):
                        update_draft_status(draft['id'], "published")
                        st.rerun()
            
            with col4:
                if st.button("🗑️ Delete", key=f"lib_delete_{draft['id']}"):
                    conn = sqlite3.connect(DATABASE_PATH)
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM content_drafts WHERE id = ?", (draft['id'],))
                    conn.commit()
                    conn.close()
                    st.rerun()
            
            # Scheduling interface
            if st.session_state.get(f"lib_scheduling_{draft['id']}"):
                schedule_date = st.date_input("Schedule for:", key=f"lib_date_{draft['id']}")
                schedule_time = st.time_input("Time:", key=f"lib_time_{draft['id']}")
                
                if st.button("Confirm Schedule", key=f"lib_confirm_{draft['id']}"):
                    scheduled_datetime = f"{schedule_date} {schedule_time}"
                    update_draft_status(draft['id'], "scheduled", scheduled_datetime)
                    st.success(f"Scheduled for {scheduled_datetime}")
                    st.session_state[f"lib_scheduling_{draft['id']}"] = False
                    st.rerun()

def content_calendar_page(user_profile: UserProfile):
    """Content calendar and posting suggestions"""
    st.title("📅 Content Calendar")
    st.markdown("View your content schedule and get posting recommendations.")
    
    # Get content counts
    approved_drafts = get_content_drafts("approved")
    scheduled_drafts = get_content_drafts("scheduled")
    
    # Posting cadence analysis
    st.subheader("📊 Content Buffer Analysis")
    
    total_ready_content = len(approved_drafts) + len(scheduled_drafts)
    cadence_suggestions = suggest_posting_cadence(user_profile.posting_frequency, total_ready_content)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Ready to Post", len(approved_drafts))
    
    with col2:
        st.metric("Scheduled", len(scheduled_drafts))
    
    with col3:
        st.metric("Total Buffer", total_ready_content)
    
    # Display recommendations
    st.markdown("### 📋 Recommendations")
    st.info(cadence_suggestions["recommendation"])
    st.write(f"**Content Buffer:** {cadence_suggestions['current_buffer']}")
    st.write(f"**Next Creation Session:** {cadence_suggestions['next_creation_date']}")
    
    # Calendar view of scheduled content
    st.subheader("📅 Scheduled Content")
    
    if scheduled_drafts:
        for draft in scheduled_drafts:
            if draft['scheduled_date']:
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.write(f"**{draft['scheduled_date'][:10]}**")
                
                with col2:
                    st.write(f"{draft['format_type'].replace('_', ' ').title()}: {draft['content'][:100]}...")
                
                with col3:
                    if st.button("Edit", key=f"cal_edit_{draft['id']}"):
                        st.session_state[f"editing_{draft['id']}"] = True
    else:
        st.info("No scheduled content. Go to Content Library to schedule your approved drafts!")
    
    # Content planning suggestions
    st.subheader("💡 Content Planning Ideas")
    
    # Get theme analysis for suggestions
    theme_counts = analyze_brand_balance()
    
    if theme_counts:
        st.write("**Based on your content themes, consider creating content about:**")
        
        # Find under-represented themes
        total_content = sum(theme_counts.values())
        underrepresented = []
        
        ideal_themes = ["leadership", "industry insights", "personal story", "strategy"]
        
        for theme in ideal_themes:
            current_percentage = (theme_counts.get(theme, 0) / total_content) * 100
            if current_percentage < 15:  # Less than 15% representation
                underrepresented.append(theme)
        
        if underrepresented:
            for theme in underrepresented[:3]:
                st.write(f"• More **{theme}** content (currently underrepresented)")
        else:
            st.write("• Your content themes are well balanced! Keep up the variety.")
    
    # Weekly planning assistant
    st.subheader("📋 This Week's Content Plan")
    
    today = datetime.datetime.now()
    week_start = today - datetime.timedelta(days=today.weekday())
    
    posts_this_week = []
    for draft in scheduled_drafts:
        if draft['scheduled_date']:
            draft_date = datetime.datetime.strptime(draft['scheduled_date'][:10], "%Y-%m-%d")
            if week_start <= draft_date < week_start + datetime.timedelta(days=7):
                posts_this_week.append(draft)
    
    if posts_this_week:
        st.write(f"**{len(posts_this_week)} posts scheduled for this week:**")
        for post in posts_this_week:
            st.write(f"• {post['scheduled_date'][:10]}: {post['format_type'].replace('_', ' ').title()}")
    else:
        st.warning("No posts scheduled for this week. Consider scheduling some content!")

def brand_gym_page():
    """Brand Gym - analysis, learning, and improvement"""
    st.title("💪 Brand Gym")
    st.markdown("Strengthen your personal brand with analysis, learning resources, and action prompts.")
    
    # Brand analysis
    st.subheader("📊 Brand Skills Diagnosis")
    
    # Analyze current content balance
    theme_counts = analyze_brand_balance()
    
    if theme_counts:
        # Display theme distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Your Content Theme Distribution:**")
            total_content = sum(theme_counts.values())
            
            for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_content) * 100
                st.write(f"• {theme.title()}: {percentage:.1f}% ({count} posts)")
        
        with col2:
            # Visual representation (simple bar chart using Streamlit)
            st.bar_chart(theme_counts)
        
        # Generate recommendations
        recommendations = generate_brand_recommendations(theme_counts)
        
        st.subheader("🎯 Brand Recommendations")
        for rec in recommendations:
            st.write(f"• {rec}")
        
        # Identify weak areas for learning suggestions
        total_content = sum(theme_counts.values())
        weak_themes = []
        
        for theme in ["leadership", "industry insights", "personal story", "strategy", "innovation"]:
            current_percentage = (theme_counts.get(theme, 0) / total_content) * 100
            if current_percentage < 10:  # Less than 10% representation
                weak_themes.append(theme)
        
        # Learning Feed
        st.subheader("📚 Learning Feed")
        st.markdown("Curated resources to strengthen your brand gaps:")
        
        if weak_themes:
            learning_suggestions = get_learning_suggestions(weak_themes)
            
            for suggestion in learning_suggestions:
                st.write(f"📖 {suggestion}")
        else:
            st.success("Your content is well-balanced across themes! Consider these general resources:")
            st.write("📖 Read 'The Trusted Advisor' by David Maister")
            st.write("📖 Follow thought leaders in your industry on LinkedIn")
            st.write("📖 Subscribe to Harvard Business Review for diverse insights")
        
        # Implementation Prompts
        st.subheader("🚀 Implementation Prompts")
        st.markdown("Ready-to-use content prompts based on your brand gaps:")
        
        if weak_themes:
            implementation_prompts = get_implementation_prompts(weak_themes)
            
            for i, prompt in enumerate(implementation_prompts, 1):
                with st.expander(f"Content Prompt #{i}"):
                    st.write(prompt)
                    
                    if st.button(f"Use This Prompt", key=f"prompt_{i}"):
                        # Store prompt in session state for easy access in voice capture
                        st.session_state.content_prompt = prompt
                        st.success("Prompt saved! Go to Voice Note Capture to record content based on this prompt.")
        else:
            st.write("🎉 Great job maintaining balanced content! Here are some advanced prompts:")
            advanced_prompts = [
                "Share a contrarian view about a popular trend in your industry",
                "Describe how you've evolved as a leader over the past year", 
                "Explain a framework you use that others might find valuable"
            ]
            
            for i, prompt in enumerate(advanced_prompts, 1):
                with st.expander(f"Advanced Prompt #{i}"):
                    st.write(prompt)
                    
                    if st.button(f"Use This Prompt", key=f"adv_prompt_{i}"):
                        st.session_state.content_prompt = prompt
                        st.success("Prompt saved! Go to Voice Note Capture to record content based on this prompt.")
    
    else:
        st.info("Create some content first to see your brand analysis!")
        st.markdown("### 🏁 Getting Started")
        st.write("1. Go to **Voice Note Capture** to record your first thoughts")
        st.write("2. Use **Content Generator** to create posts from your ideas")  
        st.write("3. Return here to see your brand analysis and get personalized recommendations")
    
    # Brand goals tracking
    st.subheader("🎯 Brand Goals")
    
    # Simple goal setting interface
    with st.expander("Set Brand Goals"):
        st.write("**What aspect of your personal brand do you want to strengthen this month?**")
        
        brand_goals = st.multiselect(
            "Select your focus areas:",
            ["Leadership Presence", "Industry Thought Leadership", "Personal Storytelling", 
             "Strategic Thinking", "Innovation Mindset", "Team Building", "Customer Focus"]
        )
        
        if st.button("Save Goals"):
            # In a real app, you'd save these to the database
            st.success("Goals saved! We'll provide targeted content suggestions based on these areas.")
    
    # Weekly brand challenge
    st.subheader("🏆 This Week's Brand Challenge")
    
    challenges = [
        "Share one personal failure and the lesson you learned",
        "Post about an industry trend you disagree with and why",
        "Tell the story behind a major decision you made recently",
        "Share a framework or process that's working well for your team",
        "Write about a book or article that changed your perspective"
    ]
    
    # Use current week to determine challenge
    week_num = datetime.datetime.now().isocalendar()[1]
    current_challenge = challenges[week_num % len(challenges)]
    
    st.write(f"**Challenge:** {current_challenge}")
    
    if st.button("Accept Challenge"):
        st.session_state.content_prompt = f"Brand Challenge: {current_challenge}"
        st.success("Challenge accepted! Go to Voice Note Capture to create content for this challenge.")

# Run the app
if __name__ == "__main__":
    main()
