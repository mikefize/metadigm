import pyparsing
pyparsing.DelimitedList = pyparsing.delimitedList

import streamlit as st
import google.generativeai as genai
import anthropic
import os
import time
import random
import re
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# --- APP CONFIG ---
st.set_page_config(page_title="The Paradigm: Director's Cut", page_icon="🎬", layout="wide")

# --- SESSION STATE ---
if "step" not in st.session_state: st.session_state.step = "setup"
if "dossier" not in st.session_state: st.session_state.dossier = None
if "attempt" not in st.session_state: st.session_state.attempt = 0
if "raw_story" not in st.session_state: st.session_state.raw_story = ""
if "final_story" not in st.session_state: st.session_state.final_story = ""
if "stats" not in st.session_state: st.session_state.stats = {"input": 0, "output": 0, "cost": 0.0}
if "seed" not in st.session_state: st.session_state.seed = "Paradigm"
if "manual_config" not in st.session_state: st.session_state.manual_config = {}

# --- MODEL DEFINITIONS ---
MODELS = {
    "Claude 4.6 Sonnet": {"name": "Claude 4.5 Sonnet", "id": "claude-sonnet-4-6", "vendor": "anthropic", "price_in": 3.00, "price_out": 15.00},
    "Claude 4.6 Opus": {"name": "Claude 4.5 Opus", "id": "claude-opus-4-6", "vendor": "anthropic", "price_in": 5.00, "price_out": 25.00},
    "Gemini 3 Pro": {"name": "Gemini 3 Pro", "id": "gemini-3-pro-preview", "vendor": "google", "price_in": 2.00, "price_out": 12.00},
    "Gemini 3 Flash": {"name": "Gemini 3 Flash", "id": "gemini-3-flash-preview", "vendor": "google", "price_in": 0.5, "price_out": 3.00}
}

CONFIG_DIR = 'config'
SCENARIO_DIR = 'scenarios'

# --- UTILS ---
def load_list(filename):
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path): return ["Generic Option"]
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def load_file_content(filepath):
    if not os.path.exists(filepath): return None
    with open(filepath, 'r', encoding='utf-8') as f: return f.read()

def extract_tag(text, tag_name):
    if not text: return ""
    match = re.search(r'\{\s*' + tag_name + r'\s*:(.*?)\}', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'(?:^|\n|\*)\s*' + tag_name + r'(?:\*\*|)\s*:\s*(.*)', text, re.IGNORECASE)
    if match: return match.group(1).strip()
    return ""

def clean_artifacts(text):
    if not text: return ""
    text = re.sub(r'\{\s*(State|Title|Summary|Scene)\s*:.*?\}', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[\s*(State|Title|Summary)\s*:.*?\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# --- API ---
def track_cost(in_tok, out_tok, model_config):
    st.session_state.stats['input'] += in_tok
    st.session_state.stats['output'] += out_tok
    c_in = (in_tok / 1_000_000) * model_config['price_in']
    c_out = (out_tok / 1_000_000) * model_config['price_out']
    st.session_state.stats['cost'] += (c_in + c_out)
    if 'cost_metric' in st.session_state:
        st.session_state.cost_metric.metric("Budget", f"${st.session_state.stats['cost']:.4f}")

def call_api(prompt, model_key, is_editor=False, max_tokens=8192):
    m_cfg = MODELS[model_key]
    
    sys_prompt = "You are a Senior Editor. Polish while preserving length." if is_editor else """
    You are a high-end novelist writing a Dark Psychological Thriller.
    RULES:
    1. **ACTIVE ANTAGONIST:** The transformation must not just "happen." It must be enforced by the Antagonist/Mechanism.
    2. **POV:** Adhere strictly to the requested Point of View.
    3. **SHOW, DON'T TELL:** Focus on sensory details and internal monologue.
    4. **NO FOG:** Describe mental changes as specific psychological/biological processes.
    """
    
    try:
        if m_cfg['vendor'] == 'anthropic':
            client = anthropic.Anthropic(api_key=st.session_state.anthropic_key, timeout=600.0)
            resp = client.messages.create(
                model=m_cfg['id'], max_tokens=max_tokens, system=sys_prompt, 
                messages=[{"role": "user", "content": prompt}]
            )
            track_cost(resp.usage.input_tokens, resp.usage.output_tokens, m_cfg)
            return resp.content[0].text
        else:
            genai.configure(api_key=st.session_state.google_key)
            model = genai.GenerativeModel(model_name=m_cfg['id'], system_instruction=sys_prompt)
            safe = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            resp = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens}, safety_settings=safe)
            if resp.usage_metadata: 
                track_cost(resp.usage_metadata.prompt_token_count, resp.usage_metadata.candidates_token_count, m_cfg)
            return resp.text
    except Exception as e:
        return f"API ERROR: {str(e)}"

# --- GENERATION LOGIC ---
def generate_dossier(seed, attempt, config):
    random.seed(f"{seed}_{attempt}")
    
    # 1. Load basic ingredients
    scenarios = [f for f in os.listdir(SCENARIO_DIR) if f.endswith('.txt')]
    theme_file = config.get('theme') or random.choice(scenarios)
    theme_content = load_file_content(os.path.join(SCENARIO_DIR, theme_file))
    theme_name = theme_file.replace('theme_', '').replace('.txt', '').replace('_', ' ').title()
    
    genre = config.get('genre') or random.choice(load_list('genres.txt'))
    job = config.get('job') or random.choice(load_list('occupations.txt'))
    mc_method = config.get('mc_method') or random.choice(load_list('mc_methods.txt'))
    pov = config.get('pov') or "First Person (I)"

    # 2. Antagonist Logic (Simplified)
    antag_selection = config.get('antagonist', 'Random')
    
    if antag_selection == "Dynamic (AI Invented)":
        antag_instr = "**ANTAGONIST:** [OPEN - AI INVENT] (Invent a unique Villain/Force that perfectly fits this specific Job and Theme)."
        antag_display_name = "Dynamic (AI)"
    elif antag_selection == "No Antagonist":
        antag_instr = "**ANTAGONIST:** [NONE]. There is no villain. The transformation happens due to an automated process, a cursed object, her own hubris, or the environment."
        antag_display_name = "None (Environment)"
    else: # "Random from List"
        random_antag = random.choice(load_list('antagonists.txt'))
        antag_instr = f"**ANTAGONIST:** {random_antag}"
        antag_display_name = random_antag

    # 3. Body Parts Logic
    b_list = load_list('body_parts.txt')
    initial_b = config.get('body_parts') or ["__RANDOM__"] * random.choice([2, 3])
    selected_b = []
    for item in initial_b:
        if item == "__RANDOM__":
            avail = [x for x in b_list if x not in selected_b]
            if avail: selected_b.append(random.choice(avail))
        else: selected_b.append(item)
    body_string = ", ".join(selected_b)

    # 4. Fetish Logic
    f_list = load_list('fetishes.txt')
    initial_f = config.get('fetishes') or ["__RANDOM__"]
    selected_f = []
    for item in initial_f:
        if item == "__RANDOM__": 
            avail = [x for x in f_list if x not in selected_f]
            if avail: selected_f.append(random.choice(avail))
        else: selected_f.append(item)
    f_string = ", ".join(selected_f)

    # 5. Archetype Logic (Always Dynamic now)
    arch_instr = "[OPEN - AI INVENT] (Invent a unique, ironic Destination Archetype that creates contrast with her Job)."

    name = f"{random.choice(load_list('names_first.txt'))} {random.choice(load_list('names_last.txt'))}"
    char = f"{name}, {random.randint(23, 45)}, {job}"

    # 6. Premise Prompt
    prompt = f"""
    TASK: Premise for a Dark Transformation novel.
    
    **INGREDIENTS:**
    - Genre: {genre}
    - POV: {pov}
    - Theme: {theme_content}
    - {antag_instr}
    - Mind Control Method: {mc_method}
    - PHYSICAL ALTERATION TARGETS: {body_string}
    - Kink: {f_string}
    - Protagonist: {char}
    - Target Archetype: {arch_instr}
    
    **INSTRUCTIONS:**
    1. **ANTAGONIST:** If Dynamic, invent a name. If NONE, describe the system/force.
    2. **DESTINATION:** Invent the specific Archetype name.
    3. **CONFLICT:** How is the {mc_method} applied to trap the {job}?
    
    **OUTPUT FORMAT (STRICT):**
    {{Antagonist: [Name/Title or "None"]}}
    {{Destination: [Archetype Name]}}
    {{Trigger: [Why she enters the situation]}}
    {{Conflict: [The trap mechanism / How she loses control]}}
    {{Blurb: [3-sentence summary]}}
    """
    
    res = call_api(prompt, st.session_state.writer_model, max_tokens=1024)
    
    return {
        "name": name, "job": job, "theme_name": theme_name, "genre": genre, 
        "fetish": f_string, "body_parts": body_string, "mc_method": mc_method, "pov": pov,
        "antagonist_name": extract_tag(res, "Antagonist") or antag_display_name, # Use AI name or fallback
        "destination": extract_tag(res, "Destination"), 
        "trigger": extract_tag(res, "Trigger"), 
        "conflict": extract_tag(res, "Conflict"), 
        "blurb": extract_tag(res, "Blurb"),
        "raw_response": res,
        "custom_note": ""
    }

# --- UI START ---
st.title("🎬 The Metamorphosis Engine")

st.sidebar.header("Settings")
st.session_state.anthropic_key = st.sidebar.text_input("Anthropic Key", type="password")
st.session_state.google_key = st.sidebar.text_input("Google Key", type="password")
st.session_state.writer_model = st.sidebar.selectbox("Writer Model", list(MODELS.keys()), index=0)
st.session_state.editor_model = st.sidebar.selectbox("Editor Model", list(MODELS.keys()), index=3)
do_editor = st.sidebar.checkbox("Enable Editor Pass", value=True)
st.session_state.cost_metric = st.sidebar.empty()
st.session_state.cost_metric.metric("Budget", f"${st.session_state.stats['cost']:.4f}")

if st.session_state.step == "setup":
    st.header("1. Production Setup")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_custom = st.checkbox("Custom Mode")
        seed = st.text_input("Story Seed", "Entropy")
        pov = st.selectbox("Point of View", ["First Person (I)", "Third Person (She)", "Second Person (You)", "Antagonist Perspective"])
        
    manual_config = {'pov': pov}
    
    if use_custom:
        with col2:
            manual_config['theme'] = st.selectbox("Theme", [None] + [f for f in os.listdir(SCENARIO_DIR)])
            manual_config['genre'] = st.selectbox("Genre", [None] + load_list('genres.txt'))
            manual_config['job'] = st.selectbox("Job", [None] + load_list('occupations.txt'))
            
            # SIMPLIFIED ANTAGONIST SELECTOR
            manual_config['antagonist'] = st.selectbox(
                "Antagonist", 
                ["Random from List", "Dynamic (AI Invented)", "No Antagonist"]
            )
            manual_config['mc_method'] = st.selectbox("MC Method", [None] + load_list('mc_methods.txt'))
            
        with col3:
            # Archetype Selector REMOVED (Always Dynamic)
            manual_config['fetishes'] = st.multiselect("Core Fetishes (Max 2)", load_list('fetishes.txt'), max_selections=2)
            manual_config['body_parts'] = st.multiselect("Physical Focus (Max 3)", load_list('body_parts.txt'), max_selections=3)

    if st.button("Draft Premise"):
        if not st.session_state.anthropic_key and not st.session_state.google_key:
            st.error("API Keys missing!")
        else:
            st.session_state.manual_config = manual_config
            st.session_state.seed = seed
            st.session_state.stats = {"input": 0, "output": 0, "cost": 0.0}
            st.session_state.cost_metric.metric("Budget", "$0.0000")
            
            with st.spinner("Drafting..."):
                d = generate_dossier(seed, st.session_state.attempt, manual_config)
                if d:
                    st.session_state.dossier = d
                    st.session_state.step = "casting"
                    st.rerun()

elif st.session_state.step == "casting":
    d = st.session_state.dossier
    st.header("2. Casting Call")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Protagonist", d['job'])
    c2.metric("Antagonist", d['antagonist_name'])
    c3.metric("MC Method", d['mc_method'])
    c4.metric("POV", d['pov'])
    
    st.subheader(f"{d['name']} -> {d.get('destination', 'Unknown')}")
    st.write(f"**Targeted Physical Changes:** {d['body_parts']}")
    
    if d['blurb']:
        st.info(f"**Conflict:** {d['conflict']}")
        st.warning(f"**Premise:** {d['blurb']}")
    else:
        st.error("Parsing Error. Raw Output:")
        st.code(d['raw_response'])

    note = st.text_area("Director's Note", placeholder="e.g. Make the ending ambiguous...")
    
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Action!"):
        st.session_state.dossier['custom_note'] = note
        st.session_state.step = "writing"
        st.rerun()
    if b2.button("🔄 Reroll"):
        st.session_state.attempt += 1
        with st.spinner("Rerolling..."):
            new_d = generate_dossier(st.session_state.seed, st.session_state.attempt, st.session_state.manual_config)
            if new_d: st.session_state.dossier = new_d
            st.rerun()
    if b3.button("❌ Back"):
        st.session_state.step = "setup"
        st.rerun()

elif st.session_state.step == "writing":
    d = st.session_state.dossier
    st.header(f"3. Filming: {d.get('destination', 'Story')}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    arc = [
        ("The Hook", "Normal life + Inciting Incident. The trap is sprung/entered."),
        ("The First Alteration", "First physical change + First MC session. She resists the process."),
        ("The Escalation", "Physical changes worsen. The coercive force tightens control."),
        ("The Mental Crack", "The 'Fog' or Logic Virus sets in. Resistance falters."),
        ("The Breaking Point", "Major event where she acts against her own original will."),
        ("Metamorphosis", "Total surrender of identity. Antagonist wins."),
        ("Epilogue", "Life in the new role.")
    ]
    
    premise = d['blurb'] if d['blurb'] else d['raw_response']
    bible = f"""
    GENRE: {d['genre']} | THEME: {d['theme_name']} | POV: {d['pov']}
    ANTAGONIST: {d['antagonist_name']} | MC METHOD: {d['mc_method']}
    ALTERATION TARGETS: {d['body_parts']} (Ensure these body parts undergo extreme transformation).
    FETISHES: {d['fetish']}
    TARGET: {d['destination']} | NOTE: {d['custom_note']}
    PREMISE: {premise} | CONFLICT: {d['conflict']}
    """
    
    full_narrative = ""
    current_state = f"Normal {d['job']}"
    raw_story = f"# {d['theme_name']}: {d.get('destination', 'Transformation')}\n\n"
    
    for i, (phase, instr) in enumerate(arc):
        status_text.write(f"Writing Chapter {i+1}: {phase}...")
        p = f"""
        {bible}
        HISTORY: {full_narrative}
        STATE: {current_state}
        TASK: Write Chapter {i+1} ({phase}). {instr}
        Use {d['pov']} perspective.
        OUTPUT: 1000 words. End with {{State: ...}} {{Title: ...}}
        """
        
        try:
            text = call_api(p, st.session_state.writer_model)
            if "API ERROR" in text:
                st.error(text)
                break
                
            current_state = extract_tag(text, "State")
            title = extract_tag(text, "Title") or phase
            clean = clean_artifacts(text)
            full_narrative += f"\n\nCHAPTER {i+1}: {title}\n{clean}"
            raw_story += f"\n\n### {title}\n\n{clean}"
            progress_bar.progress((i + 1) / (len(arc) + 1))
        except Exception as e:
            st.error(f"Error: {e}")
            break
            
    if do_editor:
        status_text.write("Editing...")
        edit_p = f"{bible}\n\nTASK: Polish manuscript. Fix logic. No summaries.\n\nINPUT:\n{raw_story}"
        try:
            final = call_api(edit_p, st.session_state.editor_model, is_editor=True, max_tokens=64000)
            st.session_state.final_story = clean_artifacts(final)
        except:
            st.session_state.final_story = clean_artifacts(raw_story)
    else:
        st.session_state.final_story = clean_artifacts(raw_story)

    progress_bar.progress(1.0)
    st.session_state.step = "final"
    st.rerun()

elif st.session_state.step == "final":
    st.header("4. Final Cut")
    st.text_area("Story", st.session_state.final_story, height=600)
    
    st.sidebar.success(f"Final Cost: ${st.session_state.stats['cost']:.4f}")
    
    safe_seed = "".join([c for c in st.session_state.seed if c.isalnum()]).rstrip()
    st.download_button("Download", st.session_state.final_story, file_name=f"Story_{safe_seed}.txt")
    
    if st.button("New Story"):
        st.session_state.step = "setup"
        st.rerun()