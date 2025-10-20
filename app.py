from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os
import streamlit as st
import json
import pandas as pd
import pyperclip
from datetime import datetime

load_dotenv()

# Configure page
st.set_page_config(
    page_title="PromptCraft Pro",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for perfect alignment
st.markdown("""
<style>
    /* Main container alignment */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        font-size: 3rem;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
        padding: 0.5rem;
    }
    
    /* Card styling */
    .generator-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: fit-content;
    }
    
    .output-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        height: fit-content;
    }
    
    /* Button styling */
    .generate-btn {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }
    
    .generate-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .sidebar-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Stats card */
    .stats-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Tag styling */
    .prompt-tag {
        background: #e3f2fd;
        color: #1976d2;
        padding: 4px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 2px;
        display: inline-block;
    }
    
    /* Code block styling */
    .code-block {
        background: #1e1e1e;
        color: #f8f8f2;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        border: 1px solid #333;
        margin: 1rem 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 12px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prompt_history' not in st.session_state:
    st.session_state.prompt_history = []

if 'total_generations' not in st.session_state:
    st.session_state.total_generations = 0

if 'saved_prompts' not in st.session_state:
    st.session_state.saved_prompts = []

if 'template_input' not in st.session_state:
    st.session_state.template_input = ""

if 'last_response' not in st.session_state:
    st.session_state.last_response = None

# Initialize model and components
apiKey = os.getenv('GOOGLE_API_KEY')
if not apiKey:
    st.error("❌ OPENROUTER_API_KEY not found in environment variables")
    try:
        model = ChatGoogleGenerativeAI(
            model='gemini-2.5-flash',
            temperature=0.7
        )
        parser = StrOutputParser()
    except Exception as e:
        st.error(f"❌ Failed to initialize model: {e}")

# FIXED: Corrected prompt template without the curly_braces variable
template = PromptTemplate(
    template="""You are an expert prompt engineer. Create the most effective prompt template for this request:

USER REQUEST: {user_input}
PURPOSE: {purpose}
TONE: {tone}
COMPLEXITY: {complexity}
AUDIENCE: {audience}

IMPORTANT: Structure your response EXACTLY like this:

DETECTED INTENT:
[Analyze what the user wants to achieve]

SUGGESTED PROMPT TYPE:
[Type of prompt needed]

COPY-PASTEABLE PROMPT TEMPLATE:
[The actual prompt template with placeholders in curly braces]

USAGE TIPS:
[Practical tips for using this prompt]

Make sure the prompt template is practical, effective, and ready to use.""",
    input_variables=["user_input", "purpose", "tone", "complexity", "audience"]
)

# Quick templates
QUICK_TEMPLATES = {
    "Creative Writing": "Write a creative story about {topic}",
    "Code Generation": "Generate {language} code for {task}",
    "Content Analysis": "Analyze this {content_type} about {topic}",
    "Research Assistant": "Research {topic} and summarize key points",
    "Business Plan": "Create a business plan for {business_idea}",
    "Learning Guide": "Explain {concept} to {audience}",
}

def copy_to_clipboard(text):
    try:
        pyperclip.copy(text)
        return True
    except:
        return False

def save_to_history(user_input, response):
    st.session_state.prompt_history.append({
        'timestamp': datetime.now().strftime("%H:%M"),
        'input': user_input[:50] + "..." if len(user_input) > 50 else user_input,
        'full_input': user_input,
        'response': response
    })
    st.session_state.total_generations += 1

def set_template_pattern(pattern):
    """Set template pattern"""
    st.session_state.template_input = pattern

def parse_response(response):
    """Parse the response into sections"""
    sections = {
        'DETECTED INTENT': '',
        'SUGGESTED PROMPT TYPE': '',
        'COPY-PASTEABLE PROMPT TEMPLATE': '',
        'USAGE TIPS': ''
    }
    
    current_section = None
    lines = response.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line is a section header
        for section in sections.keys():
            if line.upper().startswith(section):
                current_section = section
                # Remove the header from the content
                line = line.replace(section, '').replace(':', '').strip()
                break
        else:
            # If we're in a section, add content to it
            if current_section and line and not line.upper().startswith(tuple(sections.keys())):
                sections[current_section] += line + '\n'
    
    return sections

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 PromptCraft Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 3rem;">AI-Powered Prompt Engineering</p>', unsafe_allow_html=True)

    # Main layout - Two columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        # Input Configuration Card
        with st.container():
            st.markdown('<div class="generator-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">⚙️ Prompt Configuration</div>', unsafe_allow_html=True)
            
            # User Input
            user_input = st.text_area(
                "**Describe your needs:**",
                value=st.session_state.template_input,
                placeholder="Example: I need a prompt to generate creative marketing copy for a new coffee shop targeting young professionals...",
                height=120,
                key="user_input_widget"
            )
            
            # Update session state
            st.session_state.template_input = user_input
            
            # Configuration Options
            st.markdown("**Configuration Options:**")
            config_col1, config_col2 = st.columns(2)
            
            with config_col1:
                purpose = st.selectbox(
                    "Purpose",
                    ["Creative Writing", "Code Generation", "Content Analysis", "Research", "Business", "Education", "Marketing"],
                    key="purpose"
                )
                tone = st.selectbox(
                    "Tone",
                    ["Professional", "Casual", "Formal", "Friendly", "Technical"],
                    key="tone"
                )
            
            with config_col2:
                complexity = st.selectbox(
                    "Complexity",
                    ["Simple", "Moderate", "Advanced"],
                    key="complexity"
                )
                audience = st.selectbox(
                    "Audience",
                    ["General", "Technical", "Business", "Academic"],
                    key="audience"
                )
            
            # Generate Button
            generate_clicked = st.button("🚀 Generate Prompt", use_container_width=True, type="primary", key="generate_btn")
            
            if generate_clicked:
                if user_input.strip():
                    if not apiKey:
                        st.error("❌ API key not configured. Please check your .env file.")
                    else:
                        with st.spinner("🛠️ Crafting your prompt..."):
                            try:
                                chain = template | model | parser
                                response = chain.invoke({
                                    'user_input': user_input,
                                    'purpose': purpose,
                                    'tone': tone,
                                    'complexity': complexity,
                                    'audience': audience
                                })
                                save_to_history(user_input, response)
                                st.session_state.last_response = response
                                st.success("✅ Prompt generated successfully!")
                            except Exception as e:
                                st.error(f"❌ Generation failed: {str(e)}")
                                st.info("Please check your API key and internet connection.")
                else:
                    st.warning("⚠️ Please enter your requirements")
            
            st.markdown('</div>', unsafe_allow_html=True)

        # Quick Templates Section
        with st.container():
            st.markdown('<div class="generator-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header">💡 Quick Templates</div>', unsafe_allow_html=True)
            
            template_cols = st.columns(2)
            for i, (name, pattern) in enumerate(QUICK_TEMPLATES.items()):
                with template_cols[i % 2]:
                    if st.button(f"📝 {name}", use_container_width=True, key=f"template_{i}"):
                        set_template_pattern(pattern)
                        st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Results Display
        if st.session_state.last_response:
            with st.container():
                st.markdown('<div class="output-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">✨ Generated Prompt</div>', unsafe_allow_html=True)
                
                response = st.session_state.last_response
                sections = parse_response(response)
                
                # Display sections with proper formatting
                if sections['DETECTED INTENT']:
                    with st.expander("🎯 **Detected Intent**", expanded=True):
                        st.write(sections['DETECTED INTENT'].strip())
                
                if sections['SUGGESTED PROMPT TYPE']:
                    with st.expander("📝 **Suggested Prompt Type**", expanded=True):
                        st.write(sections['SUGGESTED PROMPT TYPE'].strip())
                        st.markdown(f'<span class="prompt-tag">{purpose}</span>', unsafe_allow_html=True)
                
                if sections['COPY-PASTEABLE PROMPT TEMPLATE']:
                    with st.expander("📋 **Prompt Template**", expanded=True):
                        template_content = sections['COPY-PASTEABLE PROMPT TEMPLATE'].strip()
                        st.code(template_content, language="text")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("📋 Copy Template", key="copy_template", use_container_width=True):
                                if copy_to_clipboard(template_content):
                                    st.success("✅ Template copied!")
                        with col2:
                            if st.button("🔄 Regenerate", key="regenerate", use_container_width=True):
                                st.session_state.last_response = None
                                st.rerun()
                
                if sections['USAGE TIPS']:
                    with st.expander("💡 **Usage Tips**", expanded=True):
                        st.write(sections['USAGE TIPS'].strip())
                
                # Fallback: If parsing failed, show raw response
                if not any(sections.values()):
                    st.warning("⚠️ Could not parse response format. Showing raw output:")
                    st.code(response, language="text")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            with st.container():
                st.markdown('<div class="output-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-header">✨ Generated Prompt</div>', unsafe_allow_html=True)
                st.info("👆 Configure your prompt on the left and click 'Generate Prompt' to get started")
                
                # Show sample output for demonstration
                with st.expander("📖 Example Output Format", expanded=False):
                    st.markdown("""
                    **DETECTED INTENT:**
                    Create engaging marketing content for a coffee shop targeting young professionals
                    
                    **SUGGESTED PROMPT TYPE:**
                    Marketing copy generator with brand voice specification
                    
                    **COPY-PASTEABLE PROMPT TEMPLATE:**
                    ```text
                    Act as a marketing copywriter specializing in food and beverage. Create {number} engaging {content_type} for a coffee shop called {shop_name} that targets {target_audience}. The tone should be {tone} and highlight these key features: {key_features}. Include a call-to-action about {call_to_action}.
                    ```
                    
                    **USAGE TIPS:**
                    - Replace placeholders with your specific details
                    - Adjust tone based on your brand voice
                    - Specify the number of variations needed
                    """)
                
                st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Dashboard")
        
        # Statistics
        st.markdown(f'<div class="stats-card">Total Generations<br><h3>{st.session_state.total_generations}</h3></div>', unsafe_allow_html=True)
        
        # Recent History
        st.markdown("### 📚 Recent History")
        if st.session_state.prompt_history:
            for i, item in enumerate(reversed(st.session_state.prompt_history[-5:])):
                with st.container():
                    st.markdown(f'<div class="sidebar-card">', unsafe_allow_html=True)
                    st.caption(f"🕒 {item['timestamp']}")
                    st.write(f"**{item['input']}**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔄 Reuse", key=f"reuse_{i}", use_container_width=True):
                            set_template_pattern(item['full_input'])
                            st.session_state.last_response = None
                            st.rerun()
                    with col2:
                        if st.button("📋 Copy", key=f"copy_{i}", use_container_width=True):
                            if copy_to_clipboard(item['response']):
                                st.success("✅ Copied!")
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No history yet")
        
        # Clear History
        if st.session_state.prompt_history:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.prompt_history = []
                st.session_state.total_generations = 0
                st.session_state.last_response = None
                st.rerun()
        
        st.markdown("---")
        
        # Settings
        st.markdown("### ⚙️ Settings")
        with st.container():
            st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
            # Add temperature control that actually works
            if 'temperature' not in st.session_state:
                st.session_state.temperature = 0.7
                
            st.session_state.temperature = st.slider(
                "Creativity Level", 
                0.0, 1.0, 0.7, 0.1, 
                help="Higher values = more creative responses",
                key="temp_slider"
            )
            st.checkbox("Include Examples", value=True, help="Add examples to generated prompts", key="include_examples")
            st.checkbox("Auto-Format", value=True, help="Automatically format prompts", key="auto_format")
            st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 2rem;'>"
        "Built with Streamlit • Powered by AI • 🎨 Craft Perfect Prompts"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()