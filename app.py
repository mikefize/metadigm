import pyparsing
pyparsing.DelimitedList = pyparsing.delimitedList

import streamlit as st
import google.generativeai as genai
import anthropic
import requests
import json
import os
import time
import random
import re
import warnings
from typing import Dict, Any, List, Optional, Tuple

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- APP CONFIG ---
st.set_page_config(page_title="The Paradigm: Director's Cut", page_icon="🎬", layout="wide")

# --- MODEL DEFINITIONS ---
MODELS = {
    "Grok 4.50": {"name": "Grok 4.50", "id": "grok-4.5", "vendor": "xai", "price_in": 2.00, "price_out": 6.00},
    "Grok 4.20": {"name": "Grok 4.20", "id": "grok-4.20-0309-reasoning", "vendor": "xai", "price_in": 1.25, "price_out": 2.50},
    "Claude 4.6 Sonnet": {"name": "Claude 4.5 Sonnet", "id": "claude-sonnet-4-6", "vendor": "anthropic", "price_in": 3.00, "price_out": 15.00},
    "Claude 4.6 Opus": {"name": "Claude 4.5 Opus", "id": "claude-opus-4-6", "vendor": "anthropic", "price_in": 5.00, "price_out": 25.00},
    "Gemini 3.1 Pro": {"name": "Gemini 3 Pro", "id": "gemini-3.1-pro-preview", "vendor": "google", "price_in": 2.00, "price_out": 12.00},
    "Gemini 3 Flash": {"name": "Gemini 3 Flash", "id": "gemini-3-flash-preview", "vendor": "google", "price_in": 0.50, "price_out": 3.00},
    "Gemini 3.1 Flash": {"name": "Gemini 3.1 Flash", "id": "gemini-3.1-flash-lite-preview", "vendor": "google", "price_in": 0.25, "price_out": 1.50},
    "Mistral Large": {"name": "Mistral Large", "id": "mistral-large-latest", "vendor": "mistral", "price_in": 0.50, "price_out": 1.50},
    "Kimi K3": {"name": "Kimi K3", "id": "kimi-k3", "vendor": "kimi", "price_in": 3.00, "price_out": 15.00}
}

CONFIG_DIR = 'config'

# --- COMBINATORIAL MATRIX FOR PROCEDURAL VARIANCE ---
PRESSURE_DOMAINS = [
    "High-stakes corporate audit / hostile takeover bid",
    "Undercover investigation with strict surveillance",
    "Exclusive high-society weekend retreat / gala",
    "Crucial academic / scientific grant defense deadline",
    "Isolated travel / delayed long-distance voyage",
    "High-pressure political campaign or legal trial",
    "Prestigious artistic / architectural showcase competition"
]

TICKING_CLOCKS = [
    "A strict 48-hour deadline before a public presentation",
    "An upcoming mandatory medical or security evaluation",
    "A closing window of opportunity before an opponent takes action",
    "An unannounced surprise inspection arriving at any moment",
    "A decaying environment or limited supply of a key resource"
]

RELATIONAL_DYNAMICS = [
    "Forced collaboration with a hyper-observant professional rival",
    "Tense interactions with an estranged mentor or family member",
    "Maintaining authority over skeptical subordinates who suspect something",
    "Cat-and-mouse tension with a persistent outside investigator",
    "A fragile alliance where neither party fully trusts the other"
]

SUBTEXT_CONSTRAINTS = [
    "Must conceal all physical/mental alterations under strict professional decorum",
    "Every major transformation beat must occur in a semi-public setting",
    "Dialogue must remain calm and mundane while internal panic builds",
    "The protagonist must pretend the changes are intentional personal choices",
    "Key developments must be communicated through subtext and coded language"
]

NARRATIVE_FORMATS = [
    "Linear Escalation (Chronological build)",
    "In Media Res (Opening at a late-stage crisis, then flashing back)",
    "Dual Perspective (Alternating between protagonist and an observer/antagonist)",
    "Epistolary Hybrid (Interspersed with medical logs, surveillance notes, or private memos)",
    "Pressure Cooker (Confined single-location, real-time escalation)"
]

def generate_procedural_framework() -> Dict[str, str]:
    """Generates a completely unique narrative situational framework procedurally."""
    return {
        "pressure_domain": random.choice(PRESSURE_DOMAINS),
        "ticking_clock": random.choice(TICKING_CLOCKS),
        "relational_dynamic": random.choice(RELATIONAL_DYNAMICS),
        "subtext_constraint": random.choice(SUBTEXT_CONSTRAINTS),
        "suggested_format": random.choice(NARRATIVE_FORMATS)
    }

# --- INITIALIZE SESSION STATE ---
def init_session_state():
    defaults = {
        "step": "setup",
        "seed": "Paradigm",
        "attempt": 0,
        "manual_config": {},
        "dossier": None,
        "raw_story": "",
        "final_story": "",
        "stats": {"input": 0, "output": 0, "cost": 0.0},
        "show_prompt_debug": False,
        "last_sys_prompt": "",
        "last_user_prompt": ""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session_state()

# --- UTILS ---
def load_list(filename: str) -> List[str]:
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path): 
        return ["Generic Option"]
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def load_file_content(filepath: str) -> Optional[str]:
    if not os.path.exists(filepath): 
        return None
    with open(filepath, 'r', encoding='utf-8') as f: 
        return f.read()

def extract_tag(text: str, tag_name: str) -> str:
    """Robust tag extractor handling unclosed tags and markdown fences."""
    if not text: 
        return ""
    cleaned = re.sub(r'```(?:xml|XML)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL).strip()
    
    pattern = rf'<{tag_name}>(.*?)</{tag_name}>'
    match = re.search(pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match: 
        return match.group(1).strip()
    
    pattern_unclosed = rf'<{tag_name}>(.*)'
    match_unclosed = re.search(pattern_unclosed, cleaned, re.DOTALL | re.IGNORECASE)
    if match_unclosed:
        content = match_unclosed.group(1).strip()
        next_tag = re.search(r'<[a-zA-Z0-9_]+>', content)
        if next_tag:
            content = content[:next_tag.start()].strip()
        return content

    return ""

def clean_artifacts(text: str) -> str:
    if not text: 
        return ""
    text = re.sub(r'<(state|title|summary|trigger|conflict|blurb|external_motor|narrative_format|mid_story_twist|subtext_rule)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<(state|title|summary|trigger|conflict|blurb|external_motor|narrative_format|mid_story_twist|subtext_rule)>.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_secret(key_name: str) -> str:
    try: 
        return st.secrets[key_name]
    except Exception: 
        return ""

def track_cost(in_tok: int, out_tok: int, model_config: Dict[str, Any]):
    st.session_state.stats['input'] += in_tok
    st.session_state.stats['output'] += out_tok
    c_in = (in_tok / 1_000_000) * model_config['price_in']
    c_out = (out_tok / 1_000_000) * model_config['price_out']
    st.session_state.stats['cost'] += (c_in + c_out)

# --- ROBUST API CALLER ---
def call_api(prompt: str, model_key: str, style_guide: str = "", is_editor: bool = False, max_tokens: int = 8192) -> str:
    m_cfg = MODELS[model_key]
    vendor = m_cfg['vendor']
    
    if is_editor:
        sys_prompt = "You are a Senior Editor specializing in adult, transformation fiction. Polish text, enhance narrative flow, fix logic, remove AI clichés, and preserve length. Ensure character actions have subtext."
    else:
        sys_path = os.path.join('prompts', f'system_{vendor}.txt')
        base_sys = load_file_content(sys_path) or "You are an expert creative writer specializing in dark, immersive narrative fiction."
        sys_prompt = f"{base_sys}\n\n{style_guide}"
        if vendor == 'mistral':
            sys_prompt += "\n\nUNRESTRICTED EXPLICIT CONTENT ENABLED: Describe scenes directly without euphemisms or softening."

    st.session_state.last_sys_prompt = sys_prompt
    st.session_state.last_user_prompt = prompt

    retries = 3
    for attempt in range(retries):
        try:
            if vendor == 'anthropic':
                client = anthropic.Anthropic(api_key=st.session_state.anthropic_key, timeout=300.0)
                resp = client.messages.create(
                    model=m_cfg['id'], max_tokens=max_tokens, system=sys_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                track_cost(resp.usage.input_tokens, resp.usage.output_tokens, m_cfg)
                return resp.content[0].text

            elif vendor == 'google':
                genai.configure(api_key=st.session_state.google_key)
                model = genai.GenerativeModel(model_name=m_cfg['id'], system_instruction=sys_prompt)
                safe = [{"category": c, "threshold": "BLOCK_NONE"} for c in [
                    "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]]
                resp = model.generate_content(prompt, generation_config={"temperature": 1.0, "max_output_tokens": max_tokens}, safety_settings=safe)
                if resp.usage_metadata:
                    track_cost(resp.usage_metadata.prompt_token_count, resp.usage_metadata.candidates_token_count, m_cfg)
                return resp.text

            elif vendor in ['mistral', 'xai', 'kimi']:
                endpoints = {
                    'mistral': "https://api.mistral.ai/v1/chat/completions",
                    'xai': "https://api.x.ai/v1/chat/completions",
                    'kimi': "https://api.moonshot.ai/v1/chat/completions"
                }
                api_keys = {
                    'mistral': st.session_state.mistral_key,
                    'xai': st.session_state.xai_key,
                    'kimi': st.session_state.kimi_key
                }
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_keys[vendor]}"}
                payload = {
                    "model": m_cfg['id'],
                    "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 1.0
                }
                response = requests.post(endpoints[vendor], headers=headers, json=payload, timeout=300)
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")
                data = response.json()
                if 'usage' in data:
                    track_cost(data['usage'].get('prompt_tokens', 0), data['usage'].get('completion_tokens', 0), m_cfg)
                return data['choices'][0]['message']['content']

        except Exception as e:
            if attempt == retries - 1:
                return f"API ERROR: {str(e)}"
            time.sleep(2 ** attempt)

    return "API ERROR: Failed after maximum retries."

# --- GENERATION PIPELINE ---
def generate_dossier(seed: str, attempt: int, config: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(f"{seed}_{attempt}")
    style_file = config.get('style_file', 'style_gritty.txt')
    style_guide = load_file_content(os.path.join(CONFIG_DIR, style_file)) or "Write normally."

    prots = config.get('protagonists', [{"name": "Protagonist", "gender": "Female", "info": ""}])
    prot_lines = [f"- {p['name']} (Gender: {p['gender']}" + (f", Info: {p['info']}" if p.get('info') else "") + ")" for p in prots]
    prot_str = "\n".join(prot_lines)

    antag_cfg = config.get('antagonist', {})
    if not antag_cfg.get('include', True):
        antag_str = "None"
    else:
        antag_str = f"{antag_cfg.get('name', 'AI Invented')} (Gender: {antag_cfg.get('gender', 'Female')}" + (f", Info: {antag_cfg.get('info')}" if antag_cfg.get('info') else "") + ")"

    # Body Transformation Guidance
    body_details = config.get('body_details', [])
    phys_str = "; ".join([f"{b['part']} [{b['intensity']}" + (f" ({b['remark']})" if b.get('remark') else "") + "]" for b in body_details]) if body_details else "Mental/Behavioral change focus."

    # Kink Weights Guidance
    weighted_fetishes = config.get('weighted_fetishes', {})
    f_lines = [f"- {k} (Priority Level: {v}/3)" for k, v in weighted_fetishes.items()]
    f_str = "\n".join(f_lines) if f_lines else "None specified."

    # Procedural Seed Matrix
    procedural = config.get('procedural_matrix', generate_procedural_framework())
    wildcard_level = config.get('wildcard_level', 'Dynamic Shift')

    prompt = f"""
    TASK: Architect a rich, non-formulaic transformation story dossier.

    CORE INGREDIENTS:
    - Genre: {config.get('genre', 'Open')}
    - POV: {config.get('pov', 'Third Person')}
    - Protagonists:\n{prot_str}
    - Antagonist: {antag_str}
    - Mechanism: {config.get('mc_method', 'Open')}
    - Physical Targets: {phys_str}
    - Motifs & Priorities:\n{f_str}
    - User Concept Hook: {config.get('main_idea', 'None provided.')}

    WILDCARD / UNPREDICTABILITY LEVEL: {wildcard_level}

    PROCEDURAL SEED FRAMEWORK (Use as inspiration or refine dynamically):
    - Pressure Domain: {procedural['pressure_domain']}
    - Ticking Clock: {procedural['ticking_clock']}
    - Relational Friction: {procedural['relational_dynamic']}
    - Suggested Subtext Rule: {procedural['subtext_constraint']}
    - Base Format Idea: {procedural['suggested_format']}

    DIRECTIVE FOR HIGH VARIANCE:
    1. Invent a specific, high-stakes EXTERNAL MOTOR (a real professional, social, or survival crisis beyond just 'character transforms').
    2. Invent a dramatic MID-STORY TWIST/COMPLICATION that subverts typical transformation tropes.
    3. Define a strict SUBTEXT RULE that characters must follow during dialogue and interactions.

    OUTPUT FORMAT (STRICT XML):
    <antagonist>{antag_str}</antagonist>
    <trigger>Detailed baseline personality, role, and what triggers their involvement.</trigger>
    <conflict>How the transformation mechanism unfolds and challenges their internal/external world.</conflict>
    <external_motor>Synthesized High-Stakes External Situation & Pressure Motor</external_motor>
    <mid_story_twist>Unexpected Plot Complication or Subversion</mid_story_twist>
    <narrative_format>Chosen Structural Format (e.g. In Media Res, Dual POV, Epistolary Hybrid, Linear)</narrative_format>
    <subtext_rule>Mandatory Scene Constraint / Subtext Rule</subtext_rule>
    <blurb>6-sentence comprehensive story premise integrating all factors.</blurb>
    """
    
    res = call_api(prompt, st.session_state.writer_model, style_guide=style_guide)
    if res.startswith("API ERROR"):
        return {"error": res}

    return {
        "name": prots[0]['name'] if prots else "Protagonist",
        "genre": config.get('genre'),
        "pov": config.get('pov'),
        "protagonists": prots,
        "antagonist": antag_str,
        "mc_method": config.get('mc_method'),
        "external_motor": extract_tag(res, "external_motor") or procedural['pressure_domain'],
        "mid_story_twist": extract_tag(res, "mid_story_twist") or "Unexpected external discovery.",
        "narrative_format": extract_tag(res, "narrative_format") or procedural['suggested_format'],
        "subtext_rule": extract_tag(res, "subtext_rule") or procedural['subtext_constraint'],
        "fetish_str": f_str,
        "body_details": body_details,
        "trigger": extract_tag(res, "trigger"),
        "conflict": extract_tag(res, "conflict"),
        "blurb": extract_tag(res, "blurb"),
        "style_guide": style_guide,
        "num_chapters": config.get('num_chapters', 7),
        "target_words": config.get('target_words', 10000),
        "main_idea": config.get('main_idea', ''),
        "pacing": config.get('pacing', 'Steady Build'),
        "transform_onset": config.get('transform_onset', 'Mid-Story'),
        "add_epilogue": config.get('add_epilogue', False)
    }

def generate_arc_proposal(d: Dict[str, Any], model_key: str) -> str:
    num_ch = d.get('num_chapters', 7) + (1 if d.get('add_epilogue', False) else 0)
    prompt = f"""
    TASK: Construct a unique chapter-by-chapter outline for a transformation story.
    
    PREMISE: {d.get('blurb')}
    NARRATIVE FORMAT: {d.get('narrative_format')}
    EXTERNAL MOTOR: {d.get('external_motor')}
    MID-STORY TWIST: {d.get('mid_story_twist')}
    SUBTEXT RULE: {d.get('subtext_rule')}
    PACING: {d.get('pacing')} | ONSET: {d.get('transform_onset')}
    TOTAL CHAPTERS: {num_ch}

    CRITICAL STRUCTURAL RULES:
    1. If Onset is 'Late', early chapters MUST focus entirely on the external plot motor with zero physical changes.
    2. Explicitly integrate the MID-STORY TWIST into the middle chapters.
    3. Ensure every chapter specifies a distinct scene objective, an external conflict beat, and how the SUBTEXT RULE applies.

    OUTPUT EXACT FORMAT:
    CHAPTER 1: [Title]
    - Scene Objective: ...
    - External Conflict Beat: ...
    - Transformation Beat: ...
    - Subtext Focus: ...
    CHAPTER 2: [Title]
    ...
    """
    res = call_api(prompt, model_key, max_tokens=2048)
    return clean_artifacts(res)

def build_chapter_prompt(d: Dict[str, Any], ch_idx: int, total_ch: int, ch_title: str, ch_beats: str, last_chapter_text: str, state_bible: str) -> str:
    progress_ratio = (ch_idx + 1) / total_ch
    onset = d.get('transform_onset', 'Mid-Story')
    
    # Dynamic Pacing Guardrails
    pacing_rules = "## PACING & CONSTRAINTS\n"
    if progress_ratio <= 0.35 and onset in ['Mid-Story', 'Late (Heavy Context)']:
        pacing_rules += "🛑 SETUP PHASE: Focus on mundane life, relationships, subtext, and setting up the external plot motor. DO NOT execute physical transformation yet.\n"
    elif progress_ratio >= 0.85:
        pacing_rules += "🔥 CLIMAX/METAMORPHOSIS: Full escalation. Physical and psychological surrender reaches its peak.\n"

    if ch_idx < total_ch - 1:
        pacing_rules += f"🚫 ANTI-RUSH DIRECTIVE: Chapter {ch_idx + 1} of {total_ch}. End on unresolved external or internal tension.\n"

    pacing_rules += f"🎭 MANDATORY SUBTEXT RULE: {d.get('subtext_rule', 'Conceal internal state.')}\n"

    if d.get('custom_note'):
        pacing_rules += f"🎬 DIRECTOR NOTE: {d['custom_note']}\n"

    words_per_ch = d.get('target_words', 10000) // total_ch

    return f"""
# GLOBAL STORY BIBLE
PREMISE: {d.get('blurb')}
FORMAT: {d.get('narrative_format')}
EXTERNAL MOTOR: {d.get('external_motor')}
MID-STORY TWIST: {d.get('mid_story_twist')}
MOTIFS: {d.get('fetish_str')}
POV: {d.get('pov')}

# COMPACT STORY STATE (MEMORY)
{state_bible if state_bible else "(Story is starting. Establish baseline.)"}

# IMMEDIATELY PRECEDING CHAPTER TEXT (FOR CONTINUITY)
{last_chapter_text[-3000:] if last_chapter_text else "(First chapter)"}

{pacing_rules}

# YOUR TASK
Write Chapter {ch_idx + 1}: {ch_title}.
CHAPTER SPECIFIC BEATS:
{ch_beats}

Target Word Count: ~{words_per_ch} words. Show, don't tell. Leverage subtext and character interaction.

OUTPUT AT THE VERY END OF YOUR RESPONSE:
<state_update>
- Current Physical Alterations: ...
- Current Psychological Acceptance (0-100%): ...
- Active Secrets/Relationships: ...
</state_update>
<title>{ch_title}</title>
"""

# --- UI RENDERERS ---
st.sidebar.header("⚙️ Configuration")
st.session_state.anthropic_key = st.sidebar.text_input("Anthropic Key", value=get_secret("ANTHROPIC_API_KEY"), type="password")
st.session_state.google_key = st.sidebar.text_input("Google Key", value=get_secret("GOOGLE_API_KEY"), type="password")
st.session_state.mistral_key = st.sidebar.text_input("Mistral Key", value=get_secret("MISTRAL_API_KEY"), type="password")
st.session_state.xai_key = st.sidebar.text_input("xAI (Grok) Key", value=get_secret("XAI_API_KEY"), type="password")
st.session_state.kimi_key = st.sidebar.text_input("Kimi Key", value=get_secret("KIMI_API_KEY"), type="password")

st.session_state.writer_model = st.sidebar.selectbox("Writer Model", list(MODELS.keys()), index=0)
st.session_state.editor_model = st.sidebar.selectbox("Editor Model", list(MODELS.keys()), index=2)
do_editor = st.sidebar.checkbox("Enable Editor Pass", value=True)
st.session_state.show_prompt_debug = st.sidebar.checkbox("Show Prompt Debugger", value=False)

style_files = [f for f in os.listdir(CONFIG_DIR) if f.startswith('style_') and f.endswith('.txt')] if os.path.exists(CONFIG_DIR) else []
style_choice = st.sidebar.selectbox("Writing Style Profile", style_files if style_files else ["style_gritty.txt"])

st.sidebar.metric("Budget Spent", f"${st.session_state.stats['cost']:.4f}")

# --- STEP 1: SETUP ---
if st.session_state.step == "setup":
    st.title("🎬 The Paradigm: Custom Setup")
    
    col1, col2, col3 = st.columns(3)
    manual_config = {'style_file': style_choice}

    with col1:
        st.subheader("1. Core Engine & Unpredictability")
        manual_config['seed'] = st.text_input("Seed Base", "Entropy")
        manual_config['pov'] = st.selectbox("Point of View", ["Third Person (She/He)", "First Person (I)", "Second Person (You)", "Antagonist Perspective"])
        
        manual_config['wildcard_level'] = st.select_slider(
            "Narrative Unpredictability / Wildcard Level",
            options=["Grounded (Standard)", "Dynamic Shift", "Chaotic Subversion"],
            value="Dynamic Shift",
            help="Controls how aggressively the engine invents external twists, subtext rules, and unique narrative structures."
        )

        if st.button("🎲 Procedurally Roll New Situation Matrix"):
            st.session_state.procedural_matrix = generate_procedural_framework()
            st.toast("New Procedural Matrix rolled!")

        if 'procedural_matrix' not in st.session_state:
            st.session_state.procedural_matrix = generate_procedural_framework()

        manual_config['procedural_matrix'] = st.session_state.procedural_matrix
        
        with st.expander("🔍 Inspect Current Procedural Matrix", expanded=False):
            st.json(st.session_state.procedural_matrix)

    with col2:
        st.subheader("2. Pacing & Transformation")
        manual_config['num_chapters'] = st.number_input("Number of Chapters", 3, 15, 7)
        manual_config['target_words'] = st.number_input("Target Total Words", 3000, 30000, 10000, step=1000)
        manual_config['add_epilogue'] = st.checkbox("Add Post-Transformation Epilogue", value=True)
        
        st.markdown("---")
        manual_config['pacing'] = st.select_slider("Overall Story Pacing", ["Fast & Explicit", "Steady Build", "Agonizing Slow Burn"], value="Steady Build")
        manual_config['transform_onset'] = st.select_slider("Transformation Onset", ["Chapter 1", "Mid-Story", "Late (Heavy Context)"], value="Mid-Story")

        enable_phys = st.checkbox("Physical Transformation Focus?", value=True)
        if enable_phys and os.path.exists(CONFIG_DIR):
            b_list = load_list('body_parts.txt')
            selected_b = st.multiselect("Body Focus Target Areas", b_list, max_selections=3)
            body_details = []
            for bp in selected_b:
                with st.expander(f"Focus: {bp}", expanded=True):
                    intensity = st.select_slider("Intensity", ["Subtle", "Pronounced", "Extreme"], value="Pronounced", key=f"int_{bp}")
                    remark = st.text_input("Quality Remark (e.g. natural, surgical, hyper-sensual)", key=f"rem_{bp}")
                    body_details.append({"part": bp, "intensity": intensity, "remark": remark.strip()})
            manual_config['body_details'] = body_details

    with col3:
        st.subheader("3. Cast & Core Concept")
        st.caption("Protagonist(s)")
        num_prot = st.number_input("Protagonists Count", 1, 3, 1)
        prots = []
        for i in range(num_prot):
            with st.expander(f"Protagonist {i+1}", expanded=True):
                p_name = st.text_input(f"Name #{i+1}", value="Elena" if i==0 else f"Char_{i+1}")
                p_gender = st.selectbox(f"Gender #{i+1}", ["Female", "Male", "Non-binary"])
                p_info = st.text_input(f"Info/Occupation #{i+1}", placeholder="e.g. Senior Auditor, stubborn")
                prots.append({"name": p_name, "gender": p_gender, "info": p_info})
        manual_config['protagonists'] = prots

        st.caption("Antagonist")
        inc_antag = st.checkbox("Include Antagonist", value=True)
        if inc_antag:
            a_name = st.text_input("Antagonist Name", value="Dr. Vance")
            a_gender = st.selectbox("Antagonist Gender", ["Female", "Male", "Non-binary"])
            a_info = st.text_input("Antagonist Role", value="Lead Researcher")
            manual_config['antagonist'] = {"name": a_name, "gender": a_gender, "info": a_info, "include": True}
        else:
            manual_config['antagonist'] = {"include": False}

        if os.path.exists(CONFIG_DIR):
            manual_config['genre'] = st.selectbox("Genre", load_list('genres.txt'))
            manual_config['mc_method'] = st.selectbox("Transformation Mechanism", load_list('mc_methods.txt'))

    st.subheader("4. Main Idea & Motifs")
    manual_config['main_idea'] = st.text_area("High-Level Concept / Hook (Optional)", placeholder="Leave blank to let the engine synthesize the narrative motor automatically...")
    
    if os.path.exists(CONFIG_DIR):
        f_list = load_list('fetishes.txt')
        selected_f = st.multiselect("Select Core Kinks/Motifs (Max 4)", f_list, max_selections=4)
        weighted_f = {}
        if selected_f:
            cols = st.columns(len(selected_f))
            for idx, f in enumerate(selected_f):
                with cols[idx]:
                    weighted_f[f] = st.slider(f"'{f}' Priority", 1, 3, 2, key=f"w_{f}")
        manual_config['weighted_fetishes'] = weighted_f

    if st.button("🚀 Draft Story Premise Dossier", use_container_width=True):
        st.session_state.manual_config = manual_config
        with st.spinner("Synthesizing Dynamic Dossier..."):
            dossier = generate_dossier(manual_config['seed'], st.session_state.attempt, manual_config)
            if "error" in dossier:
                st.error(dossier["error"])
            else:
                st.session_state.dossier = dossier
                st.session_state.step = "casting"
                st.rerun()

# --- STEP 2: CASTING & ARC REVIEW ---
elif st.session_state.step == "casting":
    d = st.session_state.dossier
    st.title("🎬 Step 2: Casting & Story Arc Architecture")

    colA, colB = st.columns(2)
    with colA:
        st.info(f"**Premise:** {d.get('blurb')}")
        st.markdown(f"**Trigger:** {d.get('trigger')}")
        st.markdown(f"**Conflict:** {d.get('conflict')}")
    with colB:
        st.markdown(f"⚡ **External Motor:** {d.get('external_motor')}")
        st.markdown(f"🔀 **Mid-Story Twist:** {d.get('mid_story_twist')}")
        st.markdown(f"📐 **Format:** {d.get('narrative_format')}")
        st.markdown(f"🎭 **Subtext Rule:** {d.get('subtext_rule')}")

    if 'arc_proposal' not in d:
        with st.spinner("Constructing Dynamic Chapter Outline..."):
            d['arc_proposal'] = generate_arc_proposal(d, st.session_state.writer_model)
            st.session_state.dossier = d

    st.subheader("📖 Editable Chapter Arc")
    edited_arc = st.text_area("Review and edit chapter beats before filming:", value=d.get('arc_proposal'), height=250)
    d['arc_proposal'] = edited_arc

    note = st.text_area("Director's Note (Optional specific constraints)", placeholder="e.g. Ensure character keeps a secret journal until Chapter 5.")
    d['custom_note'] = note

    c1, c2, c3 = st.columns(3)
    if c1.button("✅ Action! Begin Filming", use_container_width=True):
        st.session_state.step = "writing"
        st.rerun()
    if c2.button("🔄 Reroll Premise & Motor", use_container_width=True):
        st.session_state.attempt += 1
        st.session_state.procedural_matrix = generate_procedural_framework()
        st.session_state.manual_config['procedural_matrix'] = st.session_state.procedural_matrix
        with st.spinner("Re-synthesizing Dossier..."):
            dossier = generate_dossier(st.session_state.seed, st.session_state.attempt, st.session_state.manual_config)
            st.session_state.dossier = dossier
            st.rerun()
    if c3.button("⬅️ Back to Setup", use_container_width=True):
        st.session_state.step = "setup"
        st.rerun()

# --- STEP 3: WRITING ---
elif st.session_state.step == "writing":
    d = st.session_state.dossier
    st.title(f"🎥 Filming: {d.get('name')}")
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    # Parse Arc
    arc_lines = [l.strip() for l in d['arc_proposal'].split('\n') if l.strip()]
    chapters = []
    curr_title = ""
    curr_beats = []
    
    for line in arc_lines:
        if line.upper().startswith("CHAPTER"):
            if curr_title:
                chapters.append((curr_title, "\n".join(curr_beats)))
            curr_title = line
            curr_beats = []
        else:
            curr_beats.append(line)
    if curr_title:
        chapters.append((curr_title, "\n".join(curr_beats)))

    raw_story = f"# {d.get('name')}: Metamorphosis\n\n"
    state_bible = "Baseline normal state."
    last_chapter_text = ""

    for i, (ch_title, ch_beats) in enumerate(chapters):
        status_text.write(f"Writing Chapter {i+1}/{len(chapters)}: {ch_title}...")
        
        prompt = build_chapter_prompt(
            d=d, ch_idx=i, total_ch=len(chapters),
            ch_title=ch_title, ch_beats=ch_beats,
            last_chapter_text=last_chapter_text,
            state_bible=state_bible
        )

        res = call_api(prompt, st.session_state.writer_model, style_guide=d['style_guide'])
        if res.startswith("API ERROR"):
            st.error(res)
            break

        state_update = extract_tag(res, "state_update")
        if state_update:
            state_bible = state_update
            
        clean_text = clean_artifacts(res)
        last_chapter_text = clean_text
        raw_story += f"\n\n### {ch_title}\n\n{clean_text}"
        
        progress_bar.progress((i + 1) / (len(chapters) + 1))

    st.session_state.raw_story = raw_story

    if do_editor:
        status_text.write("Applying Senior Editor Polish Pass (Anti-AI Cliche Filter)...")
        edit_prompt = f"TASK: Polish manuscript. Fix logic, sensory details, and narrative flow.\n\n{raw_story}"
        edited = call_api(edit_prompt, st.session_state.editor_model, is_editor=True)
        st.session_state.final_story = clean_artifacts(edited) if not edited.startswith("API ERROR") else raw_story
    else:
        st.session_state.final_story = raw_story

    progress_bar.progress(1.0)
    st.session_state.step = "final"
    st.rerun()

# --- STEP 4: FINAL CUT ---
elif st.session_state.step == "final":
    st.title("🎬 Step 4: Final Cut")
    
    if st.session_state.show_prompt_debug:
        with st.expander("🔍 Prompt Debugger", expanded=False):
            st.subheader("Last System Prompt")
            st.code(st.session_state.last_sys_prompt)
            st.subheader("Last User Prompt")
            st.code(st.session_state.last_user_prompt)

    raw = st.session_state.raw_story
    final = st.session_state.final_story

    if do_editor and raw != final:
        t1, t2 = st.tabs(["✨ Edited Version", "📜 Raw Draft"])
        with t1:
            st.text_area("Final Polish", final, height=600)
            st.download_button("Download Edited Story", final, file_name=f"{st.session_state.seed}_EDITED.txt")
        with t2:
            st.text_area("Raw Unedited", raw, height=600)
            st.download_button("Download Raw Story", raw, file_name=f"{st.session_state.seed}_RAW.txt")
    else:
        st.text_area("Final Story", final, height=600)
        st.download_button("Download Story", final, file_name=f"{st.session_state.seed}.txt")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Rewrite (Same Parameters)", use_container_width=True):
            st.session_state.step = "writing"
            st.rerun()
    with col_b:
        if st.button("✨ New Story (Start Over)", use_container_width=True):
            st.session_state.step = "setup"
            st.rerun()