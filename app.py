"""Streamlit entry point: launches the VibeMatch web app.
git pull origin main
python -m streamlit run app.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml
from transformers import DistilBertTokenizerFast

'''
from src.model.classifier import GenreClassifier
from src.model.encoder import VibeMatchEncoder
from src.retrieval.engine import load_index
from src.retrieval.search import SearchResult
from src.retrieval.search import query as faiss_query
'''


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VibeMatch",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Config ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_config() -> dict:
    with open("configs/app_config.yaml") as f:
        return yaml.safe_load(f)


# ── Models ───────────────────────────────────────────────────────────────────
'''
@st.cache_resource
def load_models(cfg: dict):
    with open("configs/train_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = train_cfg["clip"]["projection_dim"]

    encoder = VibeMatchEncoder(projection_dim=dim)
    encoder.load_state_dict(torch.load(cfg["clip_weights_path"], map_location=device))
    encoder.to(device).eval()

    classifier = GenreClassifier()
    classifier.load_state_dict(torch.load(cfg["classifier_weights_path"], map_location=device))
    classifier.to(device).eval()

    index, metadata = load_index(cfg["index_path"])

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    vocab_path = Path(cfg["index_path"]).with_suffix(".bin.vocab.json")
    vocab: list[str] = json.loads(vocab_path.read_text()) if vocab_path.exists() else []

    return encoder, index, metadata, tokenizer, vocab, device, classifier


def run_search(
    text: str,
    encoder,
    classifier,
    index,
    metadata: list[dict],
    tokenizer,
    vocab: list[str],
    device: str,
    top_k: int,
) -> list[SearchResult]:
    """Encode query text, retrieve top-k results, annotate with live genre tags."""
    enc = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=32)
    with torch.no_grad():
        query_vec = encoder.encode_text(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        ).cpu().numpy().squeeze()

    results = faiss_query(index, metadata, query_vec, top_k=top_k)

    # Live genre tagging — requires pre-saved image embeddings from build_index.py
    if vocab:
        emb_path = Path("models/image_embeddings.npy")
        id_path = Path("models/image_embeddings_ids.json")
        if emb_path.exists() and id_path.exists():
            all_embs = np.load(str(emb_path))
            id_to_row = {v: i for i, v in enumerate(json.loads(id_path.read_text()))}
            for r in results:
                row_id = r.metadata.get("id", "")
                if row_id in id_to_row:
                    emb = torch.from_numpy(all_embs[id_to_row[row_id]]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        probs = torch.sigmoid(classifier(emb)).squeeze()
                    top_idx = probs.topk(3).indices.cpu().tolist()
                    r.metadata["live_genres"] = [vocab[i] for i in top_idx]

    return results

'''
# ── Global CSS ────────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter+Tight:wght@200;300;400;500;600&family=JetBrains+Mono:wght@300;400&display=swap');

:root {
  --bg-0: #0b0e17;
  --ink-0: #f5f3ee; --ink-1: #c7c4bc; --ink-2: #8a877f; --ink-3: #555250;
  --line: rgba(245,243,238,0.08); --line-2: rgba(245,243,238,0.14);
  --accent: #d4a574;
  --accent-soft: rgba(212,165,116,0.18);
  --accent-glow: rgba(212,165,116,0.55);
  --serif: 'Instrument Serif','Times New Roman',serif;
  --sans: 'Inter Tight',ui-sans-serif,sans-serif;
  --mono: 'JetBrains Mono',ui-monospace,monospace;
}

#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { display: none; }

.stApp { background-color: var(--bg-0) !important; }
.stApp > div, [data-testid="stAppViewContainer"],
[data-testid="stVerticalBlock"], .main { background: transparent !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.element-container { margin: 0 !important; }

/* Search input */

.stTextInput > div,
.stTextInput > div > div {
  border: none !important;
  background: transparent !important;
  box-shadow: none !important;
}
.stTextInput > div > div {
  border-radius: 999px !important;
  overflow: visible !important;
}
.stTextInput > div {
  overflow: visible !important;
}
[data-testid="stHorizontalBlock"] {
  overflow: visible !important;
}
[data-testid="column"] {
  overflow: visible !important;
}
.stTextInput > div > div > input {
  background: rgba(24,28,42,0.55) !important;
  border: 1px solid var(--line) !important;
  border-radius: 999px !important;
  color: var(--ink-0) !important;
  font-family: var(--serif) !important;
  font-size: 18px !important; font-style: italic !important;
  height: 58px !important;
  padding: 0 24px !important;
  box-shadow: inset 0 1px 0 rgba(245,243,238,0.06), 0 30px 80px -30px rgba(5,6,12,0.8) !important;
  backdrop-filter: blur(14px) !important;
  box-shadow: inset 0 1px 0 rgba(245,243,238,0.06), 0 30px 80px -30px rgba(5,6,12,0.8) !important;
  transition: border-color 300ms, box-shadow 300ms !important;
}
.stTextInput > div > div > input:focus {
  border-color: rgba(212,165,116,0.5) !important;
  box-shadow: inset 0 1px 0 rgba(245,243,238,0.08), 0 0 0 4px rgba(212,165,116,0.06),
              0 30px 100px -30px rgba(212,165,116,0.4) !important;
  outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--ink-3) !important; }
.stTextInput label { display: none !important; }
.stTextInput > div { border: none !important; box-shadow: none !important; background: transparent !important; }

[data-testid="stButton"] button {
  background: rgba(24,28,42,0.4) !important;
  border: 1px solid var(--line) !important;
  color: var(--ink-1) !important;
  border-radius: 999px !important;
  font-family: var(--sans) !important;
  font-size: 12px !important;
  font-weight: 300 !important;
  padding: 8px 14px !important;
  height: auto !important;
  width: 100% !important;
  box-shadow: none !important;
  backdrop-filter: blur(10px) !important;
  text-transform: none !important;
  letter-spacing: normal !important;
  transition: all 220ms !important;
}
[data-testid="stButton"] button:hover {
  border-color: rgba(212,165,116,0.4) !important;
  color: var(--ink-0) !important;
  background: rgba(30,28,20,0.45) !important;
  box-shadow: 0 0 0 1px rgba(212,165,116,0.18), 0 0 30px rgba(212,165,116,0.2) !important;
}

/* Match button 
[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:nth-child(3) [data-testid="stButton"] button {
  background: linear-gradient(180deg, #d4a574, #b8895d) !important;
  color: #1a1308 !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  letter-spacing: 0.16em !important;
  text-transform: uppercase !important;
  padding: 0 28px !important;
  height: 46px !important;
  margin: 6px 6px 0 0 !important;
  border: none !important;
  box-shadow: 0 0 0 1px rgba(220,180,130,0.4), 0 0 30px rgba(212,165,116,0.45) !important;
}
[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:nth-child(3) [data-testid="stButton"] butto:hover {
  background: linear-gradient(180deg, #e0b882, #c99a5e) !important;
  box-shadow: 0 0 0 1px rgba(220,180,130,0.6), 0 0 50px rgba(212,165,116,0.7) !important;
}

/* Card interactions */
.vm-card:hover .vm-halo  { opacity: 0.95 !important; transform: scale(1.05) !important; }
.vm-card:hover .vm-cover { transform: translateY(-6px) !important; box-shadow: 0 30px 60px -20px rgba(5,6,12,0.8) !important; }
.vm-card:hover .vm-score { opacity: 1 !important; transform: translateY(0) !important; }

@keyframes vm-shimmer { 0% { transform: translateX(-60%); } 100% { transform: translateX(160%); } }
@keyframes vm-pulse   { 0%,100% { opacity:.4; transform:scale(.85); } 50% { opacity:1; transform:scale(1); } }
</style>
"""

# ── Lights background ────────────────────────────────────────────────────

STARS_INJECTOR = """
<script>
(function(){
  var doc=window.parent.document;
  if(doc.getElementById('vm-stars'))return;

  var scene=doc.createElement('div');
  scene.id='vm-scene';
  scene.style.cssText='position:fixed;inset:0;z-index:0;pointer-events:none;background:radial-gradient(1200px 800px at 70% 10%,rgba(20,22,60,.35),transparent 60%),radial-gradient(900px 700px at 10% 90%,rgba(40,30,15,.25),transparent 60%),#0b0e17;';
  doc.body.prepend(scene);

  var cv=doc.createElement('canvas');
  cv.id='vm-stars';
  cv.style.cssText='position:fixed;inset:0;z-index:1;pointer-events:none;';
  doc.body.appendChild(cv);

  var ln=doc.createElement('div');
  ln.id='vm-lantern';
  ln.style.cssText='position:fixed;top:0;left:0;width:24px;height:24px;transform:translate3d(-200px,-200px,0);pointer-events:none;z-index:9999;';
  ln.innerHTML='<div style="position:absolute;inset:0;border-radius:50%;background:radial-gradient(circle,rgba(245,240,220,.9) 0%,rgba(235,228,210,.5) 25%,transparent 70%);filter:blur(.5px);"></div><div style="position:absolute;left:50%;top:50%;width:4px;height:4px;transform:translate(-50%,-50%);border-radius:50%;background:rgba(255,252,245,1);box-shadow:0 0 12px 2px rgba(245,240,220,.8);"></div>';
  doc.body.appendChild(ln);

  var s=doc.createElement('script');
  s.textContent='(function(){var cv=document.getElementById("vm-stars"),ln=document.getElementById("vm-lantern");if(!cv)return;var cx=cv.getContext("2d",{alpha:true}),W=0,H=0,DPR=Math.min(window.devicePixelRatio||1,2),st=[],mx={x:-9999,y:-9999,has:false},tg={x:-9999,y:-9999};function hue(){var r=Math.random();if(r<.06)return{h:60,c:.07};if(r<.12)return{h:265,c:.08};if(r<.16)return{h:200,c:.05};return{h:80,c:.012};}function build(){var N=Math.max(80,Math.min(450,Math.round(W*H/4500)));st=[];for(var i=0;i<N;i++){var z=.25+Math.pow(Math.random(),2),hc=hue();st.push({x:Math.random()*W,y:Math.random()*H,z:z,r:(.4+Math.random()*1.6)*z,ba:.18+Math.random()*.55,ts:.0006+Math.random()*.0018,tp:Math.random()*Math.PI*2,vx:(Math.random()-.5)*.018*z,vy:(Math.random()-.5)*.018*z,h:hc.h,c:hc.c});}}function resize(){W=cv.clientWidth=window.innerWidth;H=cv.clientHeight=window.innerHeight;cv.width=Math.floor(W*DPR);cv.height=Math.floor(H*DPR);cx.setTransform(DPR,0,0,DPR,0,0);build();}window.addEventListener("resize",resize);window.addEventListener("mousemove",function(e){tg.x=e.clientX;tg.y=e.clientY;mx.has=true;});window.addEventListener("mouseleave",function(){mx.has=false;tg.x=-9999;tg.y=-9999;});var t=0;function frame(){t++;if(mx.x===-9999){mx.x=tg.x;mx.y=tg.y;}mx.x+=(tg.x-mx.x)*.12;mx.y+=(tg.y-mx.y)*.12;if(ln&&mx.has){ln.style.transform="translate3d("+(mx.x-12)+"px,"+(mx.y-12)+"px,0)";}else if(ln){ln.style.transform="translate3d(-200px,-200px,0)";}cx.clearRect(0,0,W,H);if(mx.has){var g=cx.createRadialGradient(mx.x,mx.y,0,mx.x,mx.y,220);g.addColorStop(0,"rgba(245,240,220,.1)");g.addColorStop(.4,"rgba(235,225,200,.04)");g.addColorStop(1,"rgba(0,0,0,0)");cx.fillStyle=g;cx.fillRect(mx.x-240,mx.y-240,480,480);}var R2=78400;for(var i=0;i<st.length;i++){var s=st[i];s.x+=s.vx;s.y+=s.vy;if(s.x<-10)s.x=W+10;if(s.x>W+10)s.x=-10;if(s.y<-10)s.y=H+10;if(s.y>H+10)s.y=-10;var px=s.x,py=s.y;if(mx.has){px=s.x-(mx.x-W/2)*.012*s.z;py=s.y-(mx.y-H/2)*.012*s.z;}var prox=0;if(mx.has){var dx=px-mx.x,dy=py-mx.y,d2=dx*dx+dy*dy;if(d2<R2)prox=1-d2/R2;}var dim=mx.has?(.55+.45*prox):1,tw=.85+.15*Math.sin(t*s.ts*16+s.tp),alpha=Math.max(0,Math.min(1,s.ba*dim*tw+prox*.35)),r=s.r*(1+prox*.6),h=cx.createRadialGradient(px,py,0,px,py,r*6);h.addColorStop(0,"rgba(235,228,215,"+(alpha*.9)+")");h.addColorStop(.5,"rgba(220,215,200,"+(alpha*.18)+")");h.addColorStop(1,"rgba(0,0,0,0)");cx.fillStyle=h;cx.beginPath();cx.arc(px,py,r*6,0,Math.PI*2);cx.fill();cx.fillStyle="rgba(252,250,245,"+alpha+")";cx.beginPath();cx.arc(px,py,r,0,Math.PI*2);cx.fill();}requestAnimationFrame(frame);}resize();requestAnimationFrame(frame);})();';
  doc.body.appendChild(s);
})();
</script>
"""


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _encode_image(img_path: Path) -> str:
    if not img_path.exists():
        return ""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = img_path.suffix.lower().lstrip(".")
    mime = "jpeg" if ext in ("jpg", "jpeg") else ext
    return f"data:image/{mime};base64,{b64}"


def build_card_html(result: SearchResult, data_root: str = ".") -> str:
    meta = result.metadata
    title = meta.get("title", "Untitled")
    source = meta.get("source", "")
    source_label = "film" if source == "movie" else "book"
    score_pct = f"{result.score * 100:.1f}"
    dominant = meta.get("dominant_color", "#3a3d5c")

    genres = meta.get("live_genres") or [
        g.strip() for g in meta.get("genres", "").split("|") if g.strip()
    ]
    genres = genres[:3]

    halo = _hex_to_rgba(dominant, 0.6)
    data_uri = _encode_image(Path(data_root) / meta.get("image_path", ""))
    img_tag = (
        f'<img src="{data_uri}" style="position:absolute;inset:0;width:100%;'
        f'height:100%;object-fit:cover;" alt="{title}" />'
        if data_uri else ""
    )
    tags = "".join(
        f'<span style="font-family:var(--mono);font-size:9.5px;letter-spacing:.14em;'
        f'text-transform:uppercase;padding:4px 8px;border-radius:999px;'
        f'border:1px solid var(--line-2);color:var(--ink-2);'
        f'background:rgba(24,28,42,.4);">{g}</span>'
        for g in genres
    )

    return f"""
<div class="vm-card" style="position:relative;">
  <div class="vm-halo" style="position:absolute;inset:-10% -10% 6% -10%;
    border-radius:24px;filter:blur(36px);opacity:.55;z-index:0;pointer-events:none;
    background:radial-gradient(closest-side,{halo},transparent 70%);
    transition:opacity 400ms ease,transform 400ms ease;"></div>
  <div class="vm-cover" style="position:relative;aspect-ratio:2/3;border-radius:14px;
    overflow:hidden;border:1px solid var(--line);background:#181c2a;z-index:1;
    transition:transform 400ms cubic-bezier(.2,.7,.2,1),box-shadow 400ms;">
    {img_tag}
    <div style="position:absolute;inset:0;background:linear-gradient(to bottom,transparent 55%,rgba(8,10,18,.7));"></div>
    <div style="position:absolute;top:10px;left:10px;padding:3px 8px;
      border:1px solid rgba(245,243,238,.35);border-radius:999px;
      font-family:var(--mono);font-size:9px;letter-spacing:.14em;
      text-transform:uppercase;color:rgba(245,243,238,.7);">{source_label}</div>
    <div class="vm-score" style="position:absolute;top:10px;right:10px;padding:5px 9px;
      border-radius:999px;background:rgba(245,243,238,.16);backdrop-filter:blur(8px);
      border:1px solid rgba(245,243,238,.25);font-family:var(--mono);font-size:10px;
      letter-spacing:.08em;color:var(--ink-0);opacity:0;transform:translateY(-4px);
      transition:opacity 320ms,transform 320ms;">
      match <span style="color:var(--accent);font-weight:500;">{score_pct}</span>
    </div>
  </div>
  <div style="margin-top:14px;position:relative;z-index:1;">
    <div style="font-size:15px;font-weight:400;color:var(--ink-0);line-height:1.3;">
      <em style="font-family:var(--serif);font-style:italic;font-size:17px;">{title}</em>
    </div>
    <div style="margin-top:3px;font-size:12px;color:var(--ink-2);font-weight:300;">{source}</div>
    <div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:6px;">{tags}</div>
  </div>
</div>"""


def build_skeleton_html() -> str:
    skel = """
<div>
  <div style="aspect-ratio:2/3;border-radius:14px;background:rgba(24,28,42,.5);
    border:1px solid var(--line);overflow:hidden;position:relative;
    animation:vm-pulse 2.6s ease-in-out infinite;">
    <div style="position:absolute;inset:0;background:linear-gradient(115deg,
      transparent 30%,rgba(245,243,238,.05) 48%,rgba(139,159,244,.14) 52%,
      rgba(245,243,238,.05) 56%,transparent 75%);
      animation:vm-shimmer 2.6s ease-in-out infinite;"></div>
  </div>
  <div style="margin-top:14px;">
    <div style="height:9px;border-radius:4px;background:rgba(245,243,238,.08);"></div>
    <div style="height:9px;border-radius:4px;background:rgba(245,243,238,.08);width:55%;margin-top:8px;"></div>
  </div>
</div>"""
    return skel * 8


# ── Session state ─────────────────────────────────────────────────────────────

for key, default in [("results", []), ("last_query", ""), ("loading", False)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Render ────────────────────────────────────────────────────────────────────

cfg = load_config()

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
components.html(STARS_INJECTOR, height=0, scrolling=False)

st.markdown("""
<div style="position:relative;z-index:2;display:flex;align-items:center;
  justify-content:space-between;padding:28px 60px 0;color:var(--ink-2);
  font-size:12px;letter-spacing:.14em;text-transform:uppercase;">
  <span style="display:inline-flex;align-items:center;gap:10px;color:var(--ink-1);">
    <span style="width:6px;height:6px;border-radius:50%;background:#d4a574;
      box-shadow:0 0 10px 1px rgba(212,165,116,.55);"></span>
    VibeMatch
  </span>
  <nav style="display:flex;gap:28px;">
    <a href="#" style="color:var(--ink-2);text-decoration:none;">Discover</a>
    <a href="#" style="color:var(--ink-2);text-decoration:none;">About</a>
    <a href="#" style="color:var(--ink-2);text-decoration:none;">Method</a>
  </nav>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="position:relative;z-index:2;text-align:center;padding:14vh 40px 6vh;">
  <div style="font-family:var(--mono);font-size:11px;letter-spacing:.32em;
    color:var(--ink-2);text-transform:uppercase;margin-bottom:28px;">
    <span style="display:inline-block;vertical-align:middle;width:30px;height:1px;
      background:var(--line-2);margin-right:14px;"></span>
    a vibe-based discovery engine
    <span style="display:inline-block;vertical-align:middle;width:30px;height:1px;
      background:var(--line-2);margin-left:14px;"></span>
  </div>
  <h1 style="font-family:var(--serif);font-weight:300;
    font-size:clamp(64px,11vw,140px);line-height:.95;letter-spacing:-.02em;
    margin:0;color:var(--ink-0);
    text-shadow:0 0 40px rgba(235,230,215,.18),0 0 80px rgba(139,159,244,.12);">
    Find your <em style="color:#e8c490;">vibe.</em>
  </h1>
  <p style="margin:22px auto 0;max-width:540px;color:var(--ink-2);
    font-size:17px;font-weight:300;line-height:1.5;">
    Describe a mood. Discover movies and books that match.
  </p>
</div>
""", unsafe_allow_html=True)


_, col_input, col_btn, _ = st.columns([2, 6, 1, 2], gap="small")
with col_input:
    query_input = st.text_input(
        "",
        placeholder="a lonely journey through a neon-lit dystopian city",
        key="query_input",
        label_visibility="collapsed",
    )
with col_btn:
    match_clicked = st.button("Match ↗", use_container_width=True)

# Chips
EXAMPLE_PROMPTS = [
    "cozy autumn afternoon",
    "90s sci-fi paranoia",
    "sunlit Mediterranean nostalgia",
    "rainy Tokyo noir",
]

_, c1, c2, c3, c4, _ = st.columns([2, 1.5, 1.5, 1.5, 1.5, 2])
chip_cols = [c1, c2, c3, c4]
for col, prompt in zip(chip_cols, EXAMPLE_PROMPTS):
    with col:
        if st.button(prompt, key=f"chip_{prompt}"):
            st.session_state["query_input"] = prompt
            match_clicked = True
            query_input = prompt

'''
if match_clicked and query_input.strip():
    st.session_state.last_query = query_input.strip()
    with st.spinner(""):
        try:
            encoder, classifier, index, metadata, tokenizer, vocab, device = load_models(cfg)
            st.session_state.results = run_search(
                query_input.strip(), encoder, classifier,
                index, metadata, tokenizer, vocab, device,
                top_k=cfg.get("top_k", 10),
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.session_state.results = []
'''
# Results
if st.session_state.results or st.session_state.loading:
    n = len(st.session_state.results)
    state_word = "resolving" if st.session_state.loading else "matched"
    count_str = f'<span>{n} results</span>' if not st.session_state.loading else ""

    st.markdown(f"""
<div style="max-width:1240px;margin:64px auto 18px;padding:0 40px;
  display:flex;align-items:center;justify-content:space-between;
  color:var(--ink-2);font-family:var(--mono);font-size:11px;
  letter-spacing:.18em;text-transform:uppercase;position:relative;z-index:2;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:6px;height:6px;border-radius:50%;background:var(--accent);
      box-shadow:0 0 8px 1px var(--accent-glow);animation:vm-pulse 2.4s ease-in-out infinite;"></div>
    <span>{state_word}</span>
    <span style="color:var(--ink-0);font-style:italic;font-family:var(--serif);
      font-size:16px;letter-spacing:0;text-transform:none;">
      &ldquo;{st.session_state.last_query}&rdquo;
    </span>
  </div>
  {count_str}
</div>
""", unsafe_allow_html=True)

    grid_inner = (
        build_skeleton_html() if st.session_state.loading
        else "\n".join(build_card_html(r) for r in st.session_state.results)
    )

    st.markdown(f"""
<div style="max-width:1240px;margin:60px auto 80px;padding:0 40px;
  display:grid;grid-template-columns:repeat(4,1fr);gap:28px 24px;
  position:relative;z-index:2;">
  {grid_inner}
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<footer style="position:relative;z-index:2;padding:60px 40px 40px;text-align:center;
  color:var(--ink-3);font-family:var(--mono);font-size:11px;letter-spacing:.22em;text-transform:uppercase;">
  VibeMatch <span style="margin:0 10px;">·</span>
  NYU CSCI-UA Final Project <span style="margin:0 10px;">·</span> 2026
</footer>
""", unsafe_allow_html=True)


