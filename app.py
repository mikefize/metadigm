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
if "seed" not in st.session_state: st.session_state.seed = "Paradigm"
if "manual_config" not in st.session_state: st.session_state.manual_config = {}
if "stats" not in st.session_state: st.session_state.stats = {"input": 0, "output": 0, "cost": 0.0}

# --- MODEL DEFINITIONS ---
MODELS = {
    "Claude 4.6 Sonnet": {"name": "Claude 4.5 Sonnet", "id": "claude-sonnet-4-6", "vendor": "anthropic", "price_in": 3.00, "price_out": 15.00},
    "Claude 4.6 Opus": {"name": "Claude 4.5 Opus", "id": "claude-opus-4-6", "vendor": "anthropic", "price_in": 5.00, "price_out": 25.00},
    "Gemini 3.1 Pro": {"name": "Gemini 3 Pro", "id": "gemini-3.1-pro-preview", "vendor": "google", "price_in": 2.00, "price_out": 12.00},
    "Gemini 3 Flash": {"name": "Gemini 3 Flash", "id": "gemini-3-flash-preview", "vendor": "google", "price_in": 0.5, "price_out": 3.00}
}

CONFIG_DIR = 'config'

# --- NARRATIVE ARCS ---
STORY_ARCS = {
    "The Inevitable Slide": [
        ("The Hook", "Establish her sharp mind/life. The Inciting Incident traps her."),
        ("The First Alteration", "First physical/mental change. She views it with clinical horror. Rationalization."),
        ("The Escalation", "Changes accelerate. The Antagonist tightens the leash. She tries to maintain dignity."),
        ("The Fog", "Deep psychological shift. Complexity becomes painful. Simplicity becomes tempting."),
        ("The Breaking Point", "A major event forces her to act against her old morals/logic."),
        ("Metamorphosis", "Total surrender. The old ego dissolves into the new Archetype."),
        ("Epilogue", "Extensive 'Day in the Life'. Pure, happy existence in the new role.")
    ],
    "The Failed Rebellion": [
        ("The Hook", "Establish her stubborn/fighter personality. She is entrapped."),
        ("The Confrontation", "She actively argues or tries to negotiate. The Antagonist punishes her with the first change."),
        ("The Escape Attempt", "She tries to flee or sabotage the process. She fails."),
        ("The Punishment", "As a consequence of rebellion, the transformation is accelerated/intensified."),
        ("The Broken Will", "She realizes fighting is useless. The despair turns into a need for relief (submission)."),
        ("Metamorphosis", "She begs for the final change to stop the struggle. Total collapse."),
        ("Epilogue", "Extensive 'Day in the Life'. She is the most obedient of all because she was broken the hardest.")
    ],
    "The Faustian Seduction": [
        ("The Hook", "She enters voluntarily, arrogant or curious. She thinks she can handle it."),
        ("The Rush", "The first changes feel good/empowering. The fetish aspect is highly pleasurable."),
        ("The Addiction", "She seeks out more changes, ignoring the warning signs. The 'Fog' feels like a high."),
        ("The Trap Snaps", "She tries to pause or slow down, but realizes she has lost the agency to say 'Stop'."),
        ("The Hollow Shell", "Her intelligence fades, but the pleasure centers remain maxed out. Panic mixed with ecstasy."),
        ("Metamorphosis", "She willingly deletes her old self to maintain the high. Enthusiastic surrender."),
        ("Epilogue", "Extensive 'Day in the Life'. A portrait of a happy, empty vessel.")
    ]
}

# --- UTILS ---
def load_list(filename):
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path): return ["Generic Option"]
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def extract_tag(text, tag_name):
    if not text: return ""
    # XML
    match = re.search(r'<' + tag_name + r'>(.*?)</' + tag_name + r'>', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    # Brace Fallback
    match = re.search(r'\{\s*' + tag_name + r'\s*:(.*?)\}', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return ""

def clean_artifacts(text):
    if not text: return ""
    text = re.sub(r'<(state|title|summary)>.*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\{\s*(State|Title|Summary|Scene)\s*:.*?\}', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\[\s*(State|Title|Summary)\s*:.*?\]', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def get_secret(key_name):
    try: return st.secrets[key_name]
    except: return ""

# --- API ---
def track_cost(in_tok, out_tok, model_config):
    st.session_state.stats['input'] += in_tok
    st.session_state.stats['output'] += out_tok
    c_in = (in_tok / 1_000_000) * model_config['price_in']
    c_out = (out_tok / 1_000_000) * model_config['price_out']
    st.session_state.stats['cost'] += (c_in + c_out)
    if 'cost_metric' in st.session_state:
        st.session_state.cost_metric.metric("Budget", f"${st.session_state.stats['cost']:.4f}")

def call_api(prompt, model_key, is_editor=False, max_tokens=8192, timeout=600):
    m_cfg = MODELS[model_key]
    
    sys_prompt = "You are a Senior Editor. Polish while preserving length." if is_editor else """
    You are a high-end novelist writing a Dark Psychological Thriller / Erotica.
    
    **MANDATORY STYLE GUIDE:**
    1. **SLOW BURN & LENGTH:** Do not rush. Write EXPANSIVE, DETAILED chapters (1500+ words).
    2. **DEEP DIVES:** Spend paragraphs on single sensations, internal monologues, and the texture of the environment.
    3. **PSYCHOLOGICAL REALISM:** Focus on the cognitive dissonance—the smart woman watching her own mind degrade.
    4. **ACTIVE ANTAGONIST:** The transformation is enforced by the Antagonist/Mechanism.
    5. **NO FOG:** Describe mental changes as specific biological processes (synaptic pruning, chemical bliss, dissociation).
    """
    
    try:
        if m_cfg['vendor'] == 'anthropic':
            client = anthropic.Anthropic(api_key=st.session_state.anthropic_key, timeout=timeout)
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
            # Use request_options for timeout in newer genai versions if supported, otherwise rely on default
            resp = model.generate_content(prompt, generation_config={"max_output_tokens": max_tokens}, safety_settings=safe)
            
            if resp.usage_metadata: 
                track_cost(resp.usage_metadata.prompt_token_count, resp.usage_metadata.candidates_token_count, m_cfg)
            return resp.text
    except Exception as e:
        return f"API ERROR: {str(e)}"

# --- FORMATTING ---
def format_antagonist_option(x):
    if x is None: return "Random from List"
    if x == "__DYNAMIC__": return "Dynamic (AI Invented)"
    if x == "__NONE__": return "NO ANTAGONIST (System/Environment driven)"
    return x

# --- GENERATION ---
def generate_dossier(seed, attempt, config, status_container):
    random.seed(f"{seed}_{attempt}")
    
    status_container.write("🎲 Rolling Ingredients...")
    
    # 1. Base Variables
    genre = config.get('genre') or random.choice(load_list('genres.txt'))
    job = config.get('job') or random.choice(load_list('occupations.txt'))
    mc_method = config.get('mc_method') or random.choice(load_list('mc_methods.txt'))
    pov = config.get('pov') or "First Person (I)"

    antag_raw = config.get('antagonist')
    if antag_raw is None: antag_raw = random.choice(load_list('antagonists.txt'))
    
    if antag_raw == "__DYNAMIC__":
        antag_instr = "**ANTAGONIST:** [OPEN - AI INVENT] (Invent a unique Villain/Force)."
        antag_display_name = "Dynamic (AI)"
    elif antag_raw == "__NONE__":
        antag_instr = "**ANTAGONIST:** [NONE]. The transformation happens due to an automated process, a cursed object, or the environment."
        antag_display_name = "None (Environment)"
    else:
        antag_instr = f"**ANTAGONIST:** {antag_raw}"
        antag_display_name = antag_raw

    # Body Parts Logic
    if config.get('enable_physical', True):
        b_list = load_list('body_parts.txt')
        initial_b = config.get('body_parts') or ["__RANDOM__"] * random.choice([2, 3])
        selected_b = []
        for item in initial_b:
            if item == "__RANDOM__":
                avail = [x for x in b_list if x not in selected_b]
                if avail: selected_b.append(random.choice(avail))
            else: selected_b.append(item)
        body_string = ", ".join(selected_b)
    else:
        body_string = "NONE. The transformation must be strictly MENTAL, BEHAVIORAL, and WARDROBE-based only. No biological changes."

    f_list = load_list('fetishes.txt')
    initial_f = config.get('fetishes') or ["__RANDOM__"]
    selected_f = []
    for item in initial_f:
        if item == "__RANDOM__": 
            avail = [x for x in f_list if x not in selected_f]
            if avail: selected_f.append(random.choice(avail))
        else: selected_f.append(item)
    f_string = ", ".join(selected_f)

    # Theme
    if config.get('mode') == 'Director':
        theme_content = config.get('pitch', "No pitch provided.")
        theme_readable = "Director's Pitch"
        elements_string = "See Pitch."
    else:
        # Load random story elements
        custom_elements = []
        if config.get('person1'): custom_elements.append(f"Character: {config['person1']}")
        if config.get('person2'): custom_elements.append(f"Character: {config['person2']}")
        if config.get('location'): custom_elements.append(f"Location: {config['location']}")
        if config.get('idea1'): custom_elements.append(f"Concept: {config['idea1']}")
        if config.get('idea2'): custom_elements.append(f"Concept: {config['idea2']}")
        elements_string = "\n- ".join(custom_elements) if custom_elements else "None specified. Invent creative elements organically."
        theme_readable = "Random Elements"

    name = f"{random.choice(load_list('names_first.txt'))} {random.choice(load_list('names_last.txt'))}"
    char = f"{name}, {random.randint(23, 45)}, {job}"

    status_container.write(f"📝 Contacting AI ({st.session_state.writer_model})...")

    prompt = f"""
    TASK: Premise for a Dark Transformation novel.
    
    **CORE INGREDIENTS:**
    - Genre: {genre}
    - POV: {pov}
    - {antag_instr}
    - Mind Control Method: {mc_method}
    - PHYSICAL ALTERATION TARGETS: {body_string}
    - Kink/Motifs: {f_string}
    - Protagonist: {char}
    
    **MODULAR STORY ELEMENTS (Weave these into the plot):**
    - {elements_string}
    
    **INSTRUCTIONS:**
    1. **ANTAGONIST:** If Dynamic, invent a name. If NONE, state "None".
    2. **DESTINATION:** Invent the specific Archetype name she transforms into.
    
    **OUTPUT FORMAT (STRICT XML):**
    <antagonist>Name/Title or "None"</antagonist>
    <destination>Archetype Name</destination>
    <trigger>Why she enters the situation</trigger>
    <conflict>The trap mechanism / How she loses control</conflict>
    <blurb>4-sentence summary integrating the elements</blurb>
    """
    
    # Shorter timeout for premise generation to catch hangs quickly
    res = call_api(prompt, st.session_state.writer_model, max_tokens=1024)
    
    if not res or "API ERROR" in res:
        status_container.error(f"Generation Failed: {res}")
        return None

    status_container.write("✅ Parsing Dossier...")
    arc_keys = list(STORY_ARCS.keys())
    selected_arc_name = random.choice(arc_keys)
    
    return {
        "name": name, "job": job, "genre": genre, 
        "fetish": f_string, "body_parts": body_string, "mc_method": mc_method, "pov": pov,
        "antagonist": extract_tag(res, "antagonist") or antag_display_name,
        "destination": extract_tag(res, "destination"), 
        "trigger": extract_tag(res, "trigger"), 
        "conflict": extract_tag(res, "conflict"), 
        "blurb": extract_tag(res, "blurb"),
        "arc_name": selected_arc_name,
        "elements_string": elements_string,
        "raw_response": res,
        "custom_note": ""
    }

# --- UI START ---
st.title("🎬 The Metamorphosis Engine")

default_anthropic = get_secret("ANTHROPIC_API_KEY")
default_google = get_secret("GOOGLE_API_KEY")

st.sidebar.header("Settings")
st.session_state.anthropic_key = st.sidebar.text_input("Anthropic Key", value=default_anthropic, type="password")
st.session_state.google_key = st.sidebar.text_input("Google Key", value=default_google, type="password")
st.session_state.writer_model = st.sidebar.selectbox("Writer Model", list(MODELS.keys()), index=0)
st.session_state.editor_model = st.sidebar.selectbox("Editor Model", list(MODELS.keys()), index=3)
do_editor = st.sidebar.checkbox("Enable Editor Pass", value=True)

st.session_state.cost_metric = st.sidebar.empty()
st.session_state.cost_metric.metric("Budget", f"${st.session_state.stats['cost']:.4f}")

if st.session_state.step == "setup":
    st.header("1. Production Setup")
    mode = st.radio("Select Production Mode:", ["Random Run", "Custom Setup", "Director Mode"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    manual_config = {'mode': 'Random'}

    if mode == "Random Run":
        manual_config['mode'] = 'Random'
        with col1:
             seed = st.text_input("Story Seed", "Entropy")
             pov = st.selectbox("Point of View", ["First Person (I)", "Third Person (She)", "Second Person (You)", "Antagonist Perspective"])
             manual_config['pov'] = pov
        with col2:
             enable_phys = st.checkbox("Include Physical Transformations?", value=True)
             manual_config['enable_physical'] = enable_phys
        with col3:
             st.info("Everything else will be randomized based on the seed.")

    elif mode == "Custom Setup":
        manual_config['mode'] = 'Custom'
        with col1:
            st.subheader("Core Options")
            seed = st.text_input("Story Seed", "Entropy")
            pov = st.selectbox("Point of View", ["First Person (I)", "Third Person (She)", "Second Person (You)", "Antagonist Perspective"])
            genre = st.selectbox("Genre", [None] + load_list('genres.txt'), format_func=lambda x: "Random" if x is None else x)
            job = st.selectbox("Protagonist Job", [None] + load_list('occupations.txt'), format_func=lambda x: "Random" if x is None else x)
            
        with col2:
            st.subheader("Mechanics")
            antagonist = st.selectbox("Antagonist", [None, "__DYNAMIC__", "__NONE__"] + load_list('antagonists.txt'), format_func=format_antagonist_option)
            mc_method = st.selectbox("MC Method", [None] + load_list('mc_methods.txt'), format_func=lambda x: "Random" if x is None else x)
            fetishes = st.multiselect("Core Fetishes (Max 2)", load_list('fetishes.txt'), max_selections=2, help="Leave blank for random")
            
            enable_phys = st.checkbox("Include Physical Transformations?", value=True)
            if enable_phys:
                body_parts = st.multiselect("Physical Focus (Max 3)", load_list('body_parts.txt'), max_selections=3, help="Leave blank for random")
            else:
                body_parts = []

        with col3:
            st.subheader("Story Elements (Optional)")
            st.caption("Leave blank for AI to invent dynamically.")
            person1 = st.text_input("Person 1", placeholder="e.g. A jealous co-worker")
            person2 = st.text_input("Person 2", placeholder="e.g. Her naive younger sister")
            location = st.text_input("Location", placeholder="e.g. An abandoned mall")
            idea1 = st.text_input("Story Idea 1", placeholder="e.g. A cursed mirror")
            idea2 = st.text_input("Story Idea 2", placeholder="e.g. A secret cult")

        manual_config.update({
            'pov': pov, 'genre': genre, 'job': job, 'antagonist': antagonist, 
            'mc_method': mc_method, 'fetishes': fetishes, 'body_parts': body_parts, 'enable_physical': enable_phys,
            'person1': person1, 'person2': person2, 'location': location, 'idea1': idea1, 'idea2': idea2
        })

    elif mode == "Director Mode":
        manual_config['mode'] = 'Director'
        with col1:
             st.subheader("The Pitch")
             seed = st.text_input("Story Seed", "Director")
             pov = st.selectbox("Point of View", ["First Person (I)", "Third Person (She)", "Second Person (You)"])
             pitch = st.text_area("Main Story Idea / Pitch", height=150, placeholder="E.g. A strict judge is captured by a cartel...")
             
        with col2:
             st.subheader("Flavor")
             fetishes = st.multiselect("Core Fetishes (Max 2)", load_list('fetishes.txt'), max_selections=2)
             enable_phys = st.checkbox("Include Physical Transformations?", value=True)
             if enable_phys:
                 body_parts = st.multiselect("Physical Focus (Max 3)", load_list('body_parts.txt'), max_selections=3)
             else:
                 body_parts = []

        with col3:
             st.info("In Director Mode, the AI will invent the Job, Genre, and Antagonist to fit your Pitch.")

        manual_config.update({
            'pov': pov, 'pitch': pitch, 'fetishes': fetishes, 'body_parts': body_parts, 'enable_physical': enable_phys
        })

    if st.button("Draft Premise"):
        if not st.session_state.anthropic_key and not st.session_state.google_key:
            st.error("API Keys missing!")
        else:
            st.session_state.manual_config = manual_config
            st.session_state.seed = seed
            st.session_state.stats = {"input": 0, "output": 0, "cost": 0.0}
            st.session_state.cost_metric.metric("Budget", "$0.0000")
            
            # USE STATUS CONTAINER FOR FEEDBACK
            with st.status("Drafting Premise...", expanded=True) as status:
                d = generate_dossier(seed, st.session_state.attempt, manual_config, status)
                if d:
                    st.session_state.dossier = d
                    st.session_state.step = "casting"
                    status.update(label="Draft Complete!", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="Generation Failed", state="error")

elif st.session_state.step == "casting":
    d = st.session_state.dossier
    st.header("2. Casting Call")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Protagonist", d['job'])
    c2.metric("Antagonist", d['antagonist'])
    c3.metric("MC Method", d['mc_method'])
    c4.metric("POV", d['pov'])
    
    st.subheader(f"{d['name']} -> {d.get('destination', 'Unknown')}")
    st.write(f"**Targeted Physical Changes:** {d['body_parts']}")
    st.caption(f"**Narrative Arc:** {d['arc_name']}")
    
    st.markdown("---")
    
    # DEBUG EXPANDER
    with st.expander("Debug: Show Raw API Output"):
        st.code(d['raw_response'])

    if d['blurb']:
        st.info(f"**Trigger:** {d['trigger']}")
        st.info(f"**Conflict:** {d['conflict']}")
        st.warning(f"**Premise:** {d['blurb']}")
    else:
        st.error("Parsing Error. Please check the Debug expander above.")

    note = st.text_area("Director's Note (Optional)", placeholder="e.g. Make sure Idea 1 happens in Chapter 3...")
    
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Action!"):
        st.session_state.dossier['custom_note'] = note
        st.session_state.step = "writing"
        st.rerun()
    if b2.button("🔄 Reroll"):
        st.session_state.attempt += 1
        st.rerun() # Just rerun to trigger the setup-to-casting flow again? 
        # Actually, rerolling needs to re-call generate_dossier.
        # Ideally, we loop back to setup or call it here.
        # Simpler: Go back to setup, user clicks Draft again (preserving config)
    if b3.button("❌ Back"):
        st.session_state.step = "setup"
        st.rerun()

elif st.session_state.step == "writing":
    d = st.session_state.dossier
    st.header(f"3. Filming: {d.get('destination', 'Story')}")
    
    # LIVE STATUS CONTAINER FOR WRITING
    status = st.status("Initializing Writer...", expanded=True)
    progress_bar = st.progress(0)
    
    arc = STORY_ARCS[d['arc_name']]
    premise = d['blurb'] if d['blurb'] else d['raw_response']
    bible = f"""
    GENRE: {d['genre']} | POV: {d['pov']}
    ANTAGONIST: {d['antagonist']} | MC METHOD: {d['mc_method']}
    ALTERATION TARGETS: {d['body_parts']}
    FETISHES: {d['fetish']}
    TARGET ARCHETYPE: {d['destination']}
    PREMISE: {premise} | CONFLICT: {d['conflict']}
    NARRATIVE ARC: {d['arc_name']} | NOTE: {d['custom_note']}
    ELEMENTS: {d['elements_string']}
    """
    
    full_narrative = ""
    current_state = f"Normal {d['job']}"
    safe_dest = re.sub(r'[^a-zA-Z0-9 ]', '', d.get('destination', 'Transformation'))
    raw_story = f"# The Fall of {d['name']} ({safe_dest})\n\n"
    
    for i, (phase, instr) in enumerate(arc):
        status.write(f"✍️ Writing Chapter {i+1}: {phase}...")
        
        word_count_instr = "1500+ words" if phase == "Epilogue" else "1500 words"
        if phase == "Epilogue": instr += " Write a long, detailed 'Day in the Life' extended epilogue."

        p = f"""
        {bible}
        HISTORY: {full_narrative}
        STATE: {current_state}
        TASK: Write Chapter {i+1} ({phase}). {instr}
        Use {d['pov']} perspective.
        PACING DIRECTIVE: SLOW BURN. Do not summarize. Write distinct, heavy scenes.
        OUTPUT: {word_count_instr}. End with {{State: ...}} {{Title: ...}}
        """
        
        try:
            text = call_api(p, st.session_state.writer_model, max_tokens=12000)
            if not text or "API ERROR" in text:
                status.error(f"Failed at Chapter {i+1}: {text}")
                break
                
            current_state = extract_tag(text, "State")
            title = extract_tag(text, "Title") or phase
            clean = clean_artifacts(text)
            full_narrative += f"\n\nCHAPTER {i+1}: {title}\n{clean}"
            raw_story += f"\n\n### {title}\n\n{clean}"
            progress_bar.progress((i + 1) / (len(arc) + 1))
        except Exception as e:
            status.error(f"Error: {e}")
            break
            
    if do_editor:
        status.write("📝 Editing (Polishing and Checking Logic)...")
        edit_p = f"{bible}\n\nTASK: Polish manuscript. Fix logic. No summaries.\n\nINPUT:\n{raw_story}"
        try:
            final = call_api(edit_p, st.session_state.editor_model, is_editor=True, max_tokens=65000)
            if not final or "API ERROR" in final:
                st.session_state.final_story = clean_artifacts(raw_story)
            elif len(final) < (len(raw_story) * 0.75):
                st.warning("Editor cut too much text. Reverting to RAW.")
                st.session_state.final_story = clean_artifacts(raw_story)
            else:
                st.session_state.final_story = clean_artifacts(final)
        except:
            st.session_state.final_story = clean_artifacts(raw_story)
    else:
        st.session_state.final_story = clean_artifacts(raw_story)

    status.update(label="Production Complete!", state="complete", expanded=False)
    progress_bar.progress(1.0)
    st.session_state.step = "final"
    st.rerun()

elif st.session_state.step == "final":
    st.header("4. Final Cut")
    st.text_area("Story", st.session_state.final_story, height=600)
    
    st.sidebar.success(f"Final Cost: ${st.session_state.stats['cost']:.4f}")
    
    safe_seed = "".join([c for c in st.session_state.seed if c.isalnum()]).rstrip()
    
    st.download_button(
        label="Download Story (.txt)",
        data=st.session_state.final_story,
        file_name=f"{safe_seed}_Story.txt",
        mime="text/plain"
    )
    
    if st.button("Write New Story"):
        st.session_state.step = "setup"
        st.rerun()