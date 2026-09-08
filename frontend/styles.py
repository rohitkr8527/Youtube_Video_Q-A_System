CSS = r"""
<style>
:root {
    --vr-bg: #0f0f0f;
    --vr-panel: #181818;
    --vr-card: #212121;
    --vr-border: #303030;
    --vr-text: #f1f1f1;
    --vr-muted: #aaaaaa;
    --vr-red: #ff0033;
}

.stApp { background: var(--vr-bg); color: var(--vr-text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
.block-container { max-width: 1500px; padding-top: 1.4rem; padding-bottom: 3rem; }

.vr-brand { display:flex; align-items:center; gap:.7rem; font-weight:800; font-size:1.35rem; letter-spacing:-.02em; }
.vr-logo { width:34px; height:24px; background:var(--vr-red); border-radius:7px; position:relative; display:inline-block; }
.vr-logo:after { content:""; position:absolute; left:13px; top:7px; border-left:9px solid white; border-top:5px solid transparent; border-bottom:5px solid transparent; }
.vr-kicker { color: var(--vr-muted); font-size:.94rem; margin-top:.2rem; }
.vr-hero { text-align:center; padding:7vh 0 3vh; }
.vr-hero h1 { font-size:clamp(2.4rem,5vw,4.8rem); line-height:1.02; letter-spacing:-.045em; margin:0 0 1rem; }
.vr-hero p { color:var(--vr-muted); font-size:1.1rem; max-width:700px; margin:0 auto 2rem; }
.vr-video-title { font-weight:700; font-size:1.05rem; margin:.7rem 0 .15rem; line-height:1.35; }
.vr-subtle { color:var(--vr-muted); font-size:.9rem; }
.vr-section-title { font-size:1.2rem; font-weight:750; margin:.2rem 0 1rem; }
.vr-source-title { color:var(--vr-muted); font-size:.82rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; margin-top:1rem; }
.vr-source-row { display:flex; gap:.55rem; flex-wrap:wrap; margin:.55rem 0 .5rem; }
.vr-source { background:#272727; border:1px solid var(--vr-border); color:var(--vr-text)!important; padding:.42rem .68rem; border-radius:999px; text-decoration:none!important; font-size:.84rem; }
.vr-source:hover { border-color:#555; background:#303030; }
.vr-divider { height:1px; background:var(--vr-border); margin:1rem 0; }

[data-testid="stChatMessage"] { background: transparent; border: 0; padding:.6rem .1rem; }
[data-testid="stChatMessage"] p { line-height:1.62; }
[data-testid="stTextInput"] input { background:#202020; border:1px solid #3a3a3a; color:white; border-radius:14px; min-height:48px; }
[data-testid="stTextInput"] input:focus { border-color:#666; box-shadow:none; }
.stButton > button, .stFormSubmitButton > button { border-radius:999px; border:1px solid #3b3b3b; background:#272727; color:white; font-weight:650; }
.stButton > button:hover, .stFormSubmitButton > button:hover { border-color:#666; color:white; }
button[kind="primary"] { background:var(--vr-red)!important; border-color:var(--vr-red)!important; color:white!important; }
[data-testid="stTabs"] button { color:#cfcfcf; }
[data-testid="stTabs"] button[aria-selected="true"] { color:white; }
[data-testid="stAlert"] { border-radius:14px; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--vr-border)!important; border-radius:16px; background:var(--vr-panel); }

.vr-quiz-counter { color:var(--vr-muted); font-size:.9rem; margin-bottom:.4rem; }
.vr-quiz-question { font-size:1.2rem; font-weight:700; margin-bottom:.8rem; line-height:1.45; }

@media (max-width: 900px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .vr-hero { padding-top:4vh; }
}
</style>
"""
