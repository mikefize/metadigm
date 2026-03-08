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

# --- NARRATIVE ARCS (UPDATED FOR CONTRAST) ---
STORY_ARCS = {
    "The Inevitable Slide": [
        ("The Life Before", "Establish her high intelligence, serious job, and meaningful relationships (spouse/partner/colleague). Show her competence. NO TRANSFORMATION YET."),
        ("The Inciting Incident", "The trap is sprung. She enters the situation confident she can handle it. The first minor physical change occurs."),
        ("The First Slip", "Physical changes accelerate. She tries to hide them from her peers/partner. She rationalizes the first mental urge."),
        ("The Erosion", "The 'Fog' deepens. Her vocabulary simplifies. She struggles to do a basic task she used to master. Panic sets in."),
        ("The Breaking Point", "A major confrontation with her old life (a meeting, a date). She fails spectacularly and humiliated, finding solace in the 'new' way."),
        ("Metamorphosis", "Total surrender. She actively rejects her old, difficult life for the new, simple one."),
        ("Epilogue", "Three months later. Show a specific interaction with someone from Chapter 1 to highlight the devastating contrast.")
    ],
    "The Failed Rebellion": [
        ("The Life Before", "Establish her strong will, feminist values, or leadership role. Show her fighting for something important. NO TRANSFORMATION YET."),
        ("The Capture", "She is entrapped. She immediately resists/argues. The Antagonist punishes her defiance with the first humiliating change."),
        ("The Struggle", "She tries to escape or sabotage the process while her body betrays her. The physical changes make resistance harder."),
        ("The Logic Virus", "The mental conditioning begins. Her anger is chemically converted into arousal/confusion. She hates that she likes it."),
        ("The Public Fall", "She is forced to display her new self to the public or her peers. The shame breaks her spirit."),
        ("Metamorphosis", "She begs for the final erasure of her old self to stop the pain of thinking."),
        ("Epilogue", "She is now the most obedient, empty version of the Archetype. Contrast her silence with her voice in Chapter 1.")
    ],
    "The Faustian Seduction": [
        ("The Life Before", "Establish her ambition, vanity, or boredom. She is successful but unfulfilled. She wants 'more'. NO TRANSFORMATION YET."),
        ("The Offer", "She accepts the change voluntarily, thinking it's a temporary upgrade or game. The initial rush is euphoric."),
        ("The Addiction", "She craves more changes. She alienates her concerned friends/family, seeing them as 'boring' or 'jealous'."),
        ("The Trap Snaps", "She tries to stop or reverse a change, but realizes she can't. The Antagonist reveals the true cost."),
        ("The Hollow Shell", "Her intelligence fades, but the pleasure centers remain maxed out. She is terrified but addicted."),
        ("Metamorphosis", "She willingly deletes her old memories to maintain the high."),
        ("Epilogue", "She is a perfect, happy object. Show her ignoring a call/visit from her past life.")
    ]
}

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
    match = re.search(r'<' + tag_name + r'>(.*?)</' + tag_name + r'>', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'\{\s*' + tag_name + r'\s*:(.*?)\}', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    match = re.search(r'(?:^|\n)\s*(?:\*|-)?\s*(?:\*\*)?' + tag_name + r'(?:\*\*)?\s*:\s*(.*)', text, re.IGNORECASE)
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
    try:
        return st.secrets[key_name]
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

def call_api(prompt, model_key, style_guide="", is_editor=False, max_tokens=8192):
    m_cfg = MODELS[model_key]
    
    # REWRITTEN STYLE GUIDE FOR CONTRAST
    sys_prompt = "You are a Senior Editor. Polish while preserving length." if is_editor else f"""
    You are a seasoned author specializing in dark, psychological, and erotic thrillers. Your style is reminiscent of authors like Megan Abbott, Dennis Lehane, and Gillian Flynn. You are a master of creating palpable suspense, exploring moral ambiguity, and writing sophisticated, visceral prose.
    
    {style_guide}
    
    **MANDATORY RULES:**
    1. **CONTRAST IS KEY:** contrast the protagonist's current degrading state with her past competence.
    2. **NO CLINICAL TERMS:** Do NOT use words like "dopamine," "synapses." Describe the feeling (heat, fuzziness).
    3. **NO BANNED SMELLS:** Do NOT describe the smell of "ozone," "copper," or "antiseptic."
    4. **NO "FOG":** Use biological terms: dissociation, synaptic misfiring, exhaustion, confusing arousal.
    5. **SHOW, DON'T TELL:** Instead of "She felt stupid," write "The math problem she could solve yesterday now looked like alien hieroglyphs."
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
            if hasattr(resp, 'prompt_feedback') and resp.prompt_feedback.block_reason:
                return "API ERROR: Prompt Blocked by Safety Filter."
            try: text = resp.text
            except ValueError: return "API ERROR: Generation halted by Safety Filter."
            if resp.usage_metadata: track_cost(resp.usage_metadata.prompt_token_count, resp.usage_metadata.candidates_token_count, m_cfg)
            return text
    except Exception as e:
        return f"API ERROR: {str(e)}"

def format_antagonist_option(x):
    if x is None: return "Random from List"
    if x == "__DYNAMIC__": return "Dynamic (AI Invented)"
    if x == "__NONE__": return "NO ANTAGONIST (System/Environment)"
    return x

# --- GENERATION ---
def generate_dossier(seed, attempt, config):
    random.seed(f"{seed}_{attempt}")
    mode = config.get('mode', 'Random')
    
    style_file = config.get('style_file', 'style_gritty.txt')
    style_guide = load_file_content(os.path.join(CONFIG_DIR, style_file)) or "Write normally."
    
    custom_arc_text = config.get('custom_arc_text', '')
    arc_choice = config.get('arc', 'Random')
    if arc_choice == "Random":
        selected_arc_name = random.choice(list(STORY_ARCS.keys()))
        arc_instr = f"Follow the structure of: '{selected_arc_name}'"
    elif arc_choice == "Custom Arc":
        selected_arc_name = "Custom Director Arc"
        arc_instr = f"**CUSTOM NARRATIVE ARC:** {custom_arc_text}"
    else:
        selected_arc_name = arc_choice
        arc_instr = f"Follow the structure of: '{selected_arc_name}'"

    if mode == 'Director':
        pitch = config.get('pitch', '')
        genre = "OPEN - INFER FROM PITCH"
        job = "OPEN - INFER FROM PITCH" 
        antag_instr = "OPEN - INFER FROM PITCH"
        mc_method = "OPEN - INFER FROM PITCH"
        elements_string = f"**PITCH:**\n{pitch}"
        name = f"{random.choice(load_list('names_first.txt'))} {random.choice(load_list('names_last.txt'))}"
        char = f"{name}, Age {random.randint(23, 45)}, Job: Infer from Pitch"
    else:
        genre = config.get('genre') or random.choice(load_list('genres.txt'))
        job = config.get('job') or random.choice(load_list('occupations.txt'))
        mc_method = config.get('mc_method') or random.choice(load_list('mc_methods.txt'))
        
        antag_raw = config.get('antagonist')
        if antag_raw is None: antag_raw = random.choice(load_list('antagonists.txt'))
        if antag_raw == "__DYNAMIC__": antag_instr = "**ANTAGONIST:** [OPEN - AI INVENT]"
        elif antag_raw == "__NONE__": antag_instr = "**ANTAGONIST:** [NONE]"
        else: antag_instr = f"**ANTAGONIST:** {antag_raw}"

        elements = []
        if config.get('person1'): elements.append(f"Relationship: {config['person1']}")
        if config.get('person2'): elements.append(f"Relationship: {config['person2']}")
        if config.get('location'): elements.append(f"Loc: {config['location']}")
        if config.get('idea1'): elements.append(f"Idea: {config['idea1']}")
        if config.get('idea2'): elements.append(f"Idea: {config['idea2']}")
        elements_string = "\n- ".join(elements) if elements else "None."
        
        name = f"{random.choice(load_list('names_first.txt'))} {random.choice(load_list('names_last.txt'))}"
        char = f"{name}, {random.randint(23, 45)}, {job}"

    weighted_fetishes = config.get('weighted_fetishes', {})
    if not weighted_fetishes:
        f_list = load_list('fetishes.txt')
        f_pick = random.choice(f_list)
        f_string = f"- {f_pick} (Weight: Essential)"
    else:
        f_lines = []
        for f_name, weight in weighted_fetishes.items():
            if weight == 3: w_desc = "Essential. Central to the plot and transformation."
            elif weight == 2: w_desc = "Important. A recurring theme."
            else: w_desc = "Minor. A subtle background detail."
            f_lines.append(f"- {f_name} (Priority: {w_desc})")
        f_string = "\n".join(f_lines)

    if config.get('enable_physical', True):
        b_list = load_list('body_parts.txt')
        initial_b = config.get('body_parts') or ["__RANDOM__"] * random.choice([2, 3])
        selected_b = [random.choice(b_list) if i == "__RANDOM__" else i for i in initial_b]
        body_string = ", ".join(list(set(selected_b)))
    else:
        body_string = "NONE. MENTAL CHANGE ONLY."

    prompt = f"""
    TASK: Premise for a Dark Transformation novel.
    
    **CORE INGREDIENTS:**
    - Genre: {genre}
    - POV: {config.get('pov', 'First Person')}
    - {antag_instr}
    - Method: {mc_method}
    - PHYSICAL TARGETS: {body_string}
    - Protagonist: {char}
    
    **KINK/MOTIFS (With Weights):**
    {f_string}
    
    **STORY SOURCE:**
    {elements_string}
    
    **OUTPUT FORMAT (STRICT XML):**
    <antagonist>Name/Title or "None"</antagonist>
    <destination>Archetype Name (Invent a unique destination)</destination>
    <blurb>4-sentence summary</blurb>
    <trigger>Why she enters the situation</trigger>
    <conflict>The trap mechanism</conflict>
    """
    
    res = call_api(prompt, st.session_state.writer_model, style_guide, max_tokens=1024)
    if not res or "API ERROR" in res: return None
    
    final_job = job if "OPEN" not in job else "Inferred from Pitch"

    return {
        "name": name, "job": final_job, "genre": genre, 
        "fetish_str": f_string, "body_parts": body_string, "mc_method": mc_method, "pov": config.get('pov'),
        "antagonist": extract_tag(res, "antagonist"),
        "destination": extract_tag(res, "destination"), 
        "trigger": extract_tag(res, "trigger"), 
        "conflict": extract_tag(res, "conflict"), 
        "blurb": extract_tag(res, "blurb"),
        "arc_name": selected_arc_name,
        "custom_arc_text": custom_arc_text, 
        "elements_string": elements_string,
        "raw_response": res,
        "custom_note": "",
        "style_guide": style_guide
    }

# --- UI START ---
st.title("🎬 The Metamorphosis Engine")

st.sidebar.header("Settings")
st.session_state.anthropic_key = st.sidebar.text_input("Anthropic Key", value=get_secret("ANTHROPIC_API_KEY"), type="password")
st.session_state.google_key = st.sidebar.text_input("Google Key", value=get_secret("GOOGLE_API_KEY"), type="password")
st.session_state.writer_model = st.sidebar.selectbox("Writer Model", list(MODELS.keys()), index=0)
st.session_state.editor_model = st.sidebar.selectbox("Editor Model", list(MODELS.keys()), index=3)
do_editor = st.sidebar.checkbox("Enable Editor Pass", value=True)

style_files = [f for f in os.listdir(CONFIG_DIR) if f.startswith('style_') and f.endswith('.txt')]
style_choice = st.sidebar.selectbox("Writing Style Profile", style_files, format_func=lambda x: x.replace('style_', '').replace('.txt', '').title())

st.session_state.cost_metric = st.sidebar.empty()
st.session_state.cost_metric.metric("Budget", f"${st.session_state.stats['cost']:.4f}")

if st.session_state.step == "setup":
    st.header("1. Production Setup")
    mode = st.radio("Select Mode:", ["Random Run", "Custom Setup", "Director Mode"], horizontal=True)
    
    col1, col2, col3 = st.columns(3)
    manual_config = {'mode': mode, 'style_file': style_choice}

    with col1:
        st.subheader("Core")
        seed = st.text_input("Story Seed", "Entropy")
        pov = st.selectbox("Point of View", ["First Person (I)", "Third Person (She)", "Second Person (You)", "Antagonist Perspective"])
        manual_config['pov'] = pov
        
        arc_opts = ["Random"] + list(STORY_ARCS.keys()) + ["Custom Arc"]
        arc_choice = st.selectbox("Narrative Arc", arc_opts)
        manual_config['arc'] = arc_choice
        if arc_choice == "Custom Arc":
            manual_config['custom_arc_text'] = st.text_area("Plot Flow", height=100)

    with col2:
        st.subheader("Mechanics")
        if mode == "Custom Setup":
            manual_config['genre'] = st.selectbox("Genre", [None] + load_list('genres.txt'), format_func=lambda x: "Random" if x is None else x)
            manual_config['job'] = st.selectbox("Job", [None] + load_list('occupations.txt'), format_func=lambda x: "Random" if x is None else x)
            manual_config['antagonist'] = st.selectbox("Antagonist", [None, "__DYNAMIC__", "__NONE__"] + load_list('antagonists.txt'), format_func=format_antagonist_option)
            manual_config['mc_method'] = st.selectbox("MC Method", [None] + load_list('mc_methods.txt'), format_func=lambda x: "Random" if x is None else x)
        
        enable_phys = st.checkbox("Physical Changes?", value=True)
        manual_config['enable_physical'] = enable_phys
        if enable_phys:
            manual_config['body_parts'] = st.multiselect("Body Focus (Max 3)", load_list('body_parts.txt'), max_selections=3)

    with col3:
        if mode == "Director Mode":
            st.subheader("The Pitch")
            manual_config['pitch'] = st.text_area("Main Story Idea", height=150)
        elif mode == "Custom Setup":
            st.subheader("Story Elements")
            manual_config['person1'] = st.text_input("Person 1")
            manual_config['person2'] = st.text_input("Person 2")
            manual_config['location'] = st.text_input("Location")
            manual_config['idea1'] = st.text_input("Idea 1")
            manual_config['idea2'] = st.text_input("Idea 2")

    st.markdown("---")
    st.subheader("Kink & Motif Weighting (Max 4)")
    f_list = load_list('fetishes.txt')
    selected_f = st.multiselect("Select up to 4 Fetishes", f_list, max_selections=4)
    
    weighted_fetishes = {}
    if selected_f:
        fc1, fc2, fc3, fc4 = st.columns(4)
        cols = [fc1, fc2, fc3, fc4]
        for idx, f in enumerate(selected_f):
            with cols[idx]:
                weight = st.slider(f"'{f}' Importance", 1, 3, 2, key=f"w_{f}")
                weighted_fetishes[f] = weight
    manual_config['weighted_fetishes'] = weighted_fetishes

    if st.button("Draft Premise"):
        if not st.session_state.anthropic_key and not st.session_state.google_key:
            st.error("API Keys missing!")
        else:
            st.session_state.manual_config = manual_config
            st.session_state.seed = seed
            st.session_state.stats = {"input": 0, "output": 0, "cost": 0.0}
            
            with st.spinner("Drafting..."):
                d = generate_dossier(seed, st.session_state.attempt, manual_config)
                if d:
                    st.session_state.dossier = d
                    st.session_state.step = "casting"
                    st.rerun()
                else: st.error("Generation failed. Check API key/credits.")

elif st.session_state.step == "casting":
    d = st.session_state.dossier
    st.header("2. Casting Call")
    
    st.subheader(f"{d['name']} -> {d.get('destination', 'Unknown')}")
    st.caption(f"**Arc:** {d['arc_name']}")
    
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**CORE:**")
        st.markdown(f"- **Job:** {d['job']}\n- **Genre:** {d['genre']}\n- **POV:** {d['pov']}")
        st.markdown(f"- **Antagonist:** {d['antagonist']}\n- **Method:** {d['mc_method']}")
    with colB:
        st.markdown("**DETAILS:**")
        st.markdown(f"- **Physical:** {d['body_parts']}")
        st.markdown(f"**Motifs & Weights:**\n{d['fetish_str']}")
    
    st.markdown("---")
    if d['blurb']:
        st.info(f"**Trigger:** {d['trigger']}")
        st.info(f"**Conflict:** {d['conflict']}")
        st.warning(f"**Premise:** {d['blurb']}")
    else:
        st.error("Parsing Error. Raw Output:")
        st.code(d['raw_response'])

    note = st.text_area("Director's Note (Optional)", placeholder="e.g. Make sure she begs at the end...")
    
    b1, b2, b3 = st.columns(3)
    if b1.button("✅ Action!"):
        st.session_state.dossier['custom_note'] = note
        st.session_state.step = "writing"
        st.rerun()
    if b2.button("🔄 Reroll Elements"):
        st.session_state.attempt += 1
        with st.spinner("Rerolling..."):
            new_d = generate_dossier(st.session_state.seed, st.session_state.attempt, st.session_state.manual_config)
            if new_d: st.session_state.dossier = new_d
            st.rerun()
    if b3.button("❌ Back to Setup"):
        st.session_state.step = "setup"
        st.rerun()

elif st.session_state.step == "writing":
    d = st.session_state.dossier
    st.header(f"3. Filming: {d.get('destination', 'Story')}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if d['arc_name'] == "Custom Director Arc":
        base_steps = ["The Hook", "The First Alteration", "The Escalation", "The Break", "The Fall", "Metamorphosis", "Epilogue"]
        arc = [(step, f"Follow CUSTOM ARC: {d['custom_arc_text']}") for step in base_steps]
    else:
        arc = STORY_ARCS[d['arc_name']]
    
    premise = d['blurb'] if d['blurb'] else d['raw_response']
    
    bible = f"""
    GENRE: {d['genre']} | POV: {d['pov']}
    ANTAGONIST: {d['antagonist']} | MC METHOD: {d['mc_method']}
    TARGET ARCHETYPE: {d['destination']}
    PHYSICAL TARGETS: {d['body_parts']}
    MOTIFS & IMPORTANCE: \n{d['fetish_str']}
    
    PREMISE: {premise}
    CONFLICT/TRAP: {d['conflict']}
    NARRATIVE ARC: {d['arc_name']}
    NOTE: {d['custom_note']}
    """
    
    full_narrative = ""
    current_state = f"Normal {d['job']}"
    raw_story = f"# The Fall of {d['name']}\n\n"
    
    for i, (phase, instr) in enumerate(arc):
        status_text.write(f"Writing Chapter {i+1}: {phase}...")
        
        p = f"""
        {bible}
        HISTORY: {full_narrative}
        STATE: {current_state}
        TASK: Write Chapter {i+1} ({phase}). {instr}
        
        **INSTRUCTIONS:** Write 1500+ words. Focus on internal monologue. Respect Motif weights.
        OUTPUT: End with EXACTLY: <state>Current Physical/Mental State</state> <title>Chapter Title</title>
        """
        
        text = call_api(p, st.session_state.writer_model, style_guide=d['style_guide'], max_tokens=12000)
        if "API ERROR" in text:
            st.error(text)
            break
            
        current_state = extract_tag(text, "state")
        title = extract_tag(text, "title") or phase
        clean = clean_artifacts(text)
        full_narrative += f"\n\nCHAPTER {i+1}: {title}\n{clean}"
        raw_story += f"\n\n### {title}\n\n{clean}"
        progress_bar.progress((i + 1) / (len(arc) + 1))
            
    if do_editor:
        status_text.write("Editing...")
        edit_p = f"{bible}\n\nTASK: Polish manuscript. Fix logic. No summaries. Remove tags.\n\nINPUT:\n{raw_story}"
        final = call_api(edit_p, st.session_state.editor_model, is_editor=True, max_tokens=65000)
        st.session_state.final_story = clean_artifacts(final) if final and len(final) > len(raw_story)*0.7 else clean_artifacts(raw_story)
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
    st.download_button("Download", st.session_state.final_story, file_name=f"{safe_seed}.txt")
    if st.button("New Story"):
        st.session_state.step = "setup"
        st.rerun()