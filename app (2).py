import re
import json
import streamlit as st
import fitz
import plotly.graph_objects as go
import plotly.express as px
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ================================================================
#  CORTEX UI — 深黑终端风格
# ================================================================
# Check if API key is in secrets to determine sidebar state
_has_secret_key = False
try:
    _has_secret_key = bool(st.secrets.get("my_deepseek_key"))
except Exception:
    pass

st.set_page_config(
    page_title="PRISM // 文件解构助手",
    layout="wide",
    initial_sidebar_state="collapsed" if _has_secret_key else "expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

:root {
    --bg: #030303;
    --bg2: #0a0a0a;
    --grid: #1a1a1a;
    --dim: #444;
    --main: #b0b0b0;
    --hi: #ffffff;
    --mono: 'JetBrains Mono', 'Courier New', monospace;
    --sans: 'Noto Sans SC', sans-serif;
}

/* 全局覆盖 Streamlit 默认样式 */
.stApp, .main, section[data-testid="stSidebar"],
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: var(--bg) !important;
    color: var(--main) !important;
}
header[data-testid="stHeader"] { background: var(--bg) !important; }

h1, h2, h3, h4, h5, h6,
.stMarkdown, .stMarkdown p, .stCaption, .stText,
label, .stSelectbox label, .stSlider label {
    font-family: var(--mono) !important;
    color: var(--main) !important;
}

/* 输入控件 */
.stSelectbox > div > div,
.stTextInput > div > div > input,
div[data-baseweb="select"] > div {
    background-color: var(--bg2) !important;
    border: 1px solid var(--grid) !important;
    color: var(--hi) !important;
    font-family: var(--mono) !important;
    border-radius: 0 !important;
}
div[data-baseweb="select"] span {
    color: var(--main) !important;
}

/* 按钮 */
.stButton > button {
    background: var(--bg2) !important;
    border: 1px solid var(--grid) !important;
    color: var(--hi) !important;
    font-family: var(--mono) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border-radius: 0 !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: var(--hi) !important;
    color: var(--bg) !important;
}

/* checkbox */
.stCheckbox label span {
    color: var(--main) !important;
    font-family: var(--mono) !important;
}

/* tabs */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg) !important;
    border-bottom: 1px solid var(--grid) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: var(--bg) !important;
    color: var(--dim) !important;
    font-family: var(--mono) !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 11px !important;
    border-right: 1px solid var(--grid) !important;
    border-radius: 0 !important;
    padding: 8px 20px !important;
}
.stTabs [aria-selected="true"] {
    color: var(--hi) !important;
    border-bottom: 1px solid var(--hi) !important;
}

/* expander */
.streamlit-expanderHeader {
    background: var(--bg2) !important;
    color: var(--main) !important;
    font-family: var(--mono) !important;
    border: 1px solid var(--grid) !important;
    border-radius: 0 !important;
}
details {
    border: 1px solid var(--grid) !important;
    border-radius: 0 !important;
}

/* progress */
.stProgress > div > div > div {
    background-color: var(--hi) !important;
}

/* slider */
section[data-testid="stSidebar"] {
    background: var(--bg) !important;
    border-right: 1px solid var(--grid) !important;
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stMarkdown {
    color: var(--dim) !important;
}

/* divider */
hr { border-color: var(--grid) !important; }

/* ---- 自定义组件 ---- */

.ctx-header {
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--grid);
    padding: 12px 0;
    margin-bottom: 20px;
    font-family: var(--mono);
}
.ctx-header .sys { color: var(--dim); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; }
.ctx-header .val { color: var(--hi); font-weight: 400; font-size: 10px; }
.ctx-title {
    font-family: var(--mono); font-size: 18px; font-weight: 700;
    color: var(--hi); letter-spacing: 2px; text-transform: uppercase;
}
.ctx-subtitle {
    font-family: var(--sans); font-size: 12px;
    color: var(--dim); margin-top: 2px;
}
.ctx-blink { animation: ctxblink 1.2s linear infinite; }
@keyframes ctxblink { 50% { opacity: 0; } }

/* 引导面板 */
.ctx-guide {
    border: 1px solid var(--grid);
    padding: 24px;
    margin: 20px 0;
    background: var(--bg2);
    font-family: var(--mono);
}
.ctx-guide-title {
    color: var(--hi); font-size: 13px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 1px;
    margin-bottom: 16px;
}
.ctx-guide-step {
    display: flex; align-items: baseline; gap: 12px;
    padding: 8px 0;
    border-bottom: 1px solid var(--grid);
    color: var(--main); font-size: 11px;
}
.ctx-guide-step:last-child { border-bottom: none; }
.ctx-guide-num {
    color: var(--hi); font-weight: 700; font-size: 12px;
    min-width: 20px;
}
.ctx-guide-active { color: var(--hi); }
.ctx-guide-done { color: var(--dim); text-decoration: line-through; }

/* 目录 */
.toc-ch {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--hi);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 8px 0 4px 0;
    margin-top: 12px;
    border-bottom: 1px solid var(--grid);
}
.toc-sec {
    font-family: var(--sans);
    font-size: 11px;
    color: var(--main);
    padding: 4px 0 4px 12px;
    border-left: 1px solid var(--grid);
    margin-left: 4px;
}
.toc-pg {
    font-family: var(--mono);
    color: var(--dim);
    font-size: 10px;
}

/* 统计 */
.ctx-stats {
    display: grid; grid-template-columns: repeat(4,1fr);
    border: 1px solid var(--grid);
}
.ctx-stat {
    text-align: center; padding: 14px 8px;
    border-right: 1px solid var(--grid);
}
.ctx-stat:last-child { border-right: none; }
.ctx-stat-n {
    font-family: var(--mono); font-size: 1.8em; font-weight: 700;
    color: var(--hi); line-height: 1;
}
.ctx-stat-l {
    font-family: var(--mono); font-size: 9px;
    color: var(--dim); text-transform: uppercase;
    letter-spacing: 1px; margin-top: 6px;
}

/* 图例 */
.ctx-lgd {
    display: flex; gap: 16px; padding: 8px 0;
    font-family: var(--mono); font-size: 9px; color: var(--dim);
    text-transform: uppercase; letter-spacing: 0.5px;
}
.ctx-lgd-d {
    display: inline-block; width: 8px; height: 8px;
    margin-right: 4px; vertical-align: middle;
}

/* 卡片 */
.kp-card {
    border-left: 2px solid;
    padding: 12px 16px;
    margin-bottom: 8px;
    background: var(--bg2);
    border-top: 1px solid var(--grid);
    border-right: 1px solid var(--grid);
    border-bottom: 1px solid var(--grid);
}
.kp-card.t-S { border-left-color: var(--hi); }
.kp-card.t-A { border-left-color: #888; }
.kp-card.t-B { border-left-color: #555; }
.kp-card.t-C { border-left-color: #333; }
.kp-card.t-D { border-left-color: var(--grid); }

.kp-head {
    display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px;
}
.kp-tier {
    font-family: var(--mono); font-size: 9px; font-weight: 700;
    padding: 1px 5px; letter-spacing: 1px;
    border: 1px solid var(--dim); color: var(--hi);
}
.kp-name {
    font-family: var(--sans); font-size: 12px; font-weight: 500;
    color: var(--hi);
}
.kp-def {
    font-family: var(--sans); font-size: 11px;
    color: var(--main); line-height: 1.7;
    margin: 4px 0 8px 0;
}
.kp-tags { display: flex; gap: 4px; flex-wrap: wrap; }
.kp-t {
    font-family: var(--mono); font-size: 9px;
    padding: 2px 6px; border: 1px solid var(--grid);
    color: var(--dim); text-transform: uppercase;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# 顶栏
st.markdown("""
<div class="ctx-header">
    <div>
        <div class="ctx-title">📖 CORTEX 知识提取器</div>
        <div class="ctx-subtitle">上传 PDF → 生成目录 → 选择章节 → 权重分析</div>
    </div>
    <div>
        <span class="sys">SYS</span> <span class="val">CORTEX_EXTRACT</span>
        &nbsp;&nbsp;
        <span class="ctx-blink">►</span> <span class="val">READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
#  Session State
# ================================================================
for k, v in {
    "vector_db":None, "page_texts":{}, "framework":"",
    "toc_entries":[], "llm":None, "processing_done":False,
    "knowledge_points":None, "raw_detail":"",
    "mastered":{}, "exercises":"", "current_query":"",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================
#  API Key 检测
# ================================================================
api_key = None
try:
    api_key = st.secrets["my_deepseek_key"]
except Exception:
    pass

# ================================================================
#  侧边栏（高级参数 + 备用 API 输入）
# ================================================================
with st.sidebar:
    st.markdown("#### PARAMETERS")
    if api_key:
        st.caption("API: CONNECTED")
    else:
        api_key = st.text_input("API_KEY", type="password")
    base_url = st.text_input("BASE_URL", value="https://api.deepseek.com")
    model_name = st.selectbox("MODEL", ["deepseek-chat","deepseek-reasoner"])
    st.markdown("---")
    toc_pages = st.slider("TOC_SCAN", 5, 30, 15)
    chunk_size = st.slider("CHUNK_SZ", 500, 2000, 1000, step=100)
    retrieve_k = st.slider("RETRIEVE_K", 4, 30, 15)
    max_chars = st.slider("BATCH_MAX", 4000, 30000, 12000, step=2000)

# ================================================================
#  核心函数
# ================================================================
@st.cache_resource
def load_emb():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    if st.session_state.llm is None:
        st.session_state.llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    return st.session_state.llm

def process_pdf(file, toc_n, csz):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    toc, pts = "", {}
    chunks, metas = [], []
    sp = RecursiveCharacterTextSplitter(chunk_size=csz, chunk_overlap=150)
    emb = load_emb()
    for i, pg in enumerate(doc):
        t = pg.get_text(); pn = i+1; pts[pn] = t
        if i < toc_n: toc += f"\n--- 第{pn}页 ---\n{t}"
        for c in sp.split_text(t):
            chunks.append(c); metas.append({"page":pn})
    vdb = FAISS.from_texts(chunks, emb, metadatas=metas)
    return vdb, pts, toc, len(doc)

def get_pages(s, e):
    pt = st.session_state.page_texts
    return "\n\n".join([f"[第{p}页]\n{pt[p]}" for p in range(s,e+1) if p in pt])

def sem_search(q, k):
    ds = st.session_state.vector_db.similarity_search(q, k=k)
    ds.sort(key=lambda d: d.metadata.get("page",0))
    return "\n".join([f"[p.{d.metadata.get('page','?')}] {d.page_content}" for d in ds])

# ================================================================
#  目录解析
# ================================================================
def parse_toc_text(toc_text):
    lines = toc_text.strip().split("\n")
    entries = []
    current_ch = None
    for line in lines:
        line = line.strip()
        if not line: continue
        pm = re.findall(r"p\.?\s*(\d+)", line)
        ps = int(pm[0]) if len(pm)>=1 else 0
        pe = int(pm[1]) if len(pm)>=2 else ps
        label = re.sub(r"\(?\s*p\.?\s*\d+\s*-?\s*p?\.?\s*\d*\s*\)?","",line).strip()
        label = re.sub(r"\s+"," ",label).strip(" -·")
        is_ch = bool(re.match(r"^(第\s*\d+\s*章|Chapter\s*\d+)", line, re.IGNORECASE))
        if is_ch:
            current_ch = {"label":label,"ps":ps,"pe":pe,"children":[]}
            entries.append(current_ch)
        else:
            sec = {"label":label,"ps":ps,"pe":pe}
            if current_ch:
                current_ch["children"].append(sec)
                if pe > current_ch["pe"]: current_ch["pe"] = pe
            else:
                entries.append({"label":label,"ps":ps,"pe":pe,"children":[]})
    return entries

def render_toc_html(entries):
    h = ""
    for ch in entries:
        h += f'<div class="toc-ch">{ch["label"]} <span class="toc-pg">p.{ch["ps"]}–{ch["pe"]}</span></div>'
        for sec in ch.get("children",[]):
            h += f'<div class="toc-sec">{sec["label"]} <span class="toc-pg">p.{sec["ps"]}–{sec["pe"]}</span></div>'
    return h

def build_opts(entries):
    opts = []
    for ch in entries:
        opts.append({"display":f"[CH] {ch['label']}  p.{ch['ps']}–{ch['pe']}",
                     "label":ch["label"],"ps":ch["ps"],"pe":ch["pe"]})
        for sec in ch.get("children",[]):
            opts.append({"display":f"  └ {sec['label']}  p.{sec['ps']}–{sec['pe']}",
                         "label":sec["label"],"ps":sec["ps"],"pe":sec["pe"]})
    return opts

# ================================================================
#  JSON 解析
# ================================================================
def robust_json(text):
    try:
        d = json.loads(text)
        if isinstance(d,list) and d: return d
    except: pass
    c = re.sub(r"```json\s*\n?","",text)
    c = re.sub(r"```\s*\n?","",c).strip()
    s,e = c.find("["),c.rfind("]")
    if s!=-1 and e>s:
        cand = c[s:e+1]
        try:
            d = json.loads(cand)
            if isinstance(d,list) and d: return d
        except: pass
        fx = re.sub(r",\s*([}\]])",r"\1",cand).replace("'",'"')
        try:
            d = json.loads(fx)
            if isinstance(d,list) and d: return d
        except: pass
    objs = []
    for m in re.finditer(r"\{[^{}]*\}",c):
        try:
            o = json.loads(m.group())
            if "name" in o: objs.append(o)
        except: continue
    return objs if objs else None

# ================================================================
#  提取
# ================================================================
EXTRACT_PROMPT = """你是一位严谨的教研专家。用户是初学者，想深度学习【{query}】。

以下是原文{batch_note}（方括号内为页码）。

■ 语言：整体中文，专业术语用「中文名 (English Term)」。
■ 权重（1-5）：core核心度, difficulty难度, connectivity关联度
■ 输出：只输出 JSON 数组，直接 [ 开头 ] 结尾，不要其他文字。

每个元素：
{{"id":"KP-01","name":"稳定匹配 (Stable Matching)","category":"所属主题","definition":"中文精准定义2-4句","page":"p.23","weights":{{"core":4,"difficulty":3,"connectivity":5,"exam_weight":4}},"prerequisites":["KP-XX"],"related":["KP-XX"]}}

原文：
{text}"""

def extract_single(text, query, llm, bl=None):
    bn = f"（第{bl}部分）" if bl else ""
    resp = llm.invoke(EXTRACT_PROMPT.format(query=query, batch_note=bn, text=text)).content
    p = robust_json(resp)
    return p if p else resp

def batch_extract(ft, query, llm, mc):
    if len(ft) <= mc: return extract_single(ft, query, llm)
    pages = ft.split("\n\n")
    batches, cur = [], ""
    for b in pages:
        if len(cur)+len(b)>mc and cur: batches.append(cur); cur=b
        else: cur += ("\n\n"+b) if cur else b
    if cur: batches.append(cur)
    all_kps = []
    prog = st.progress(0, text="EXTRACTING...")
    for i, batch in enumerate(batches):
        prog.progress((i+1)/(len(batches)+1), text=f"BATCH {i+1}/{len(batches)}")
        r = extract_single(batch, query, llm, f"{i+1}/{len(batches)}")
        if isinstance(r, list): all_kps.extend(r)
    prog.progress(1.0, text="MERGING...")
    if not all_kps: prog.empty(); return "NO_DATA"
    rp = f"""合并以下知识点去重。只输出 JSON 数组，直接 [ 开头 ] 结尾。
保持中文，专业名词「中文 (English)」。
{json.dumps(all_kps, ensure_ascii=False)}"""
    resp = llm.invoke(rp).content; prog.empty()
    p = robust_json(resp)
    return p if p else all_kps

# ================================================================
#  可视化（CORTEX 配色）
# ================================================================
def tier(c):
    if c>=5: return "S"
    if c>=4: return "A"
    if c>=3: return "B"
    if c>=2: return "C"
    return "D"

def card_html(kp):
    w = kp.get("weights",{}); c = w.get("core",3); t = tier(c)
    return f"""<div class="kp-card t-{t}">
<div class="kp-head"><span class="kp-tier">{t}</span><span class="kp-name">{kp.get('id','')}  {kp.get('name','')}</span></div>
<div class="kp-def">{kp.get('definition','—')}</div>
<div class="kp-tags">
<span class="kp-t">core {c}/5</span>
<span class="kp-t">diff {w.get('difficulty','?')}/5</span>
<span class="kp-t">conn {w.get('connectivity','?')}/5</span>
<span class="kp-t">exam {w.get('exam_weight','?')}/5</span>
<span class="kp-t">{kp.get('page','')}</span>
</div></div>"""

CTX_COLORS = [[0,"#0a0a0a"],[0.3,"#333"],[0.6,"#777"],[1,"#fff"]]

def make_treemap(kps):
    names,parents,vals,cols,hvs = [],[],[],[],[]
    cats = {}
    for kp in kps: cats.setdefault(kp.get("category","—"),[]).append(kp)
    root = "ROOT"
    names.append(root); parents.append(""); vals.append(0); cols.append(0); hvs.append("")
    for cat, items in cats.items():
        names.append(cat); parents.append(root); vals.append(0); cols.append(0)
        hvs.append(f"{cat} [{len(items)}]")
        for kp in items:
            w = kp.get("weights",{})
            s = sum(w.get(k,3) for k in ["core","difficulty","connectivity","exam_weight"])
            names.append(kp["name"]); parents.append(cat); vals.append(s); cols.append(w.get("core",3))
            hvs.append(f"{kp['name']}<br>C:{w.get('core',0)} D:{w.get('difficulty',0)} N:{w.get('connectivity',0)} E:{w.get('exam_weight',0)}")
    fig = go.Figure(go.Treemap(
        labels=names,parents=parents,values=vals,
        marker=dict(colors=cols,colorscale=CTX_COLORS,line=dict(width=1,color="#1a1a1a")),
        hovertext=hvs,hoverinfo="text",textinfo="label",
        textfont=dict(size=11,family="JetBrains Mono, monospace",color="#b0b0b0"),
        pathbar=dict(visible=True,textfont=dict(size=10,color="#444"))))
    fig.update_layout(margin=dict(t=16,l=4,r=4,b=4),height=400,
                      paper_bgcolor="#030303",font=dict(family="JetBrains Mono"))
    return fig

def make_radar(kp):
    w = kp.get("weights",{})
    cs = ["CORE","DIFF","CONN","EXAM"]
    vs = [w.get("core",0),w.get("difficulty",0),w.get("connectivity",0),w.get("exam_weight",0)]
    vs.append(vs[0])
    fig = go.Figure(go.Scatterpolar(
        r=vs,theta=cs+[cs[0]],fill="toself",
        fillcolor="rgba(255,255,255,0.03)",line=dict(color="#fff",width=1),
        marker=dict(size=4,color="#fff")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,5],tickvals=[1,2,3,4,5],
                                   tickfont=dict(size=8,color="#333"),gridcolor="#1a1a1a"),
                   angularaxis=dict(tickfont=dict(size=9,color="#555")),bgcolor="#030303"),
        showlegend=False,margin=dict(t=8,b=8,l=36,r=36),height=200,
        font=dict(family="JetBrains Mono"),paper_bgcolor="#030303")
    return fig

def make_bubble(kps):
    data = []
    for kp in kps:
        w = kp.get("weights",{})
        data.append({"name":kp["name"],"core":w.get("core",3),"diff":w.get("difficulty",3),
                     "exam":w.get("exam_weight",3),"total":w.get("core",3)+w.get("exam_weight",3)})
    fig = px.scatter(data,x="core",y="diff",size="total",color="exam",
                     hover_name="name",size_max=26,
                     color_continuous_scale=[[0,"#333"],[0.5,"#888"],[1,"#fff"]],
                     labels={"core":"CORE","diff":"DIFFICULTY","exam":"EXAM_WT"})
    fig.update_layout(height=340,margin=dict(t=16,b=16),
                      font=dict(family="JetBrains Mono",color="#888"),
                      paper_bgcolor="#030303",plot_bgcolor="#0a0a0a",
                      xaxis=dict(range=[0.5,5.5],dtick=1,gridcolor="#1a1a1a",color="#555"),
                      yaxis=dict(range=[0.5,5.5],dtick=1,gridcolor="#1a1a1a",color="#555"))
    return fig

# ================================================================
#  Prompt
# ================================================================
PROMPT_TOC = """你是图书目录整理专家。以下是书籍前若干页内容（已标注页码）。

■ 语言：章节标题翻译中文，英文原标题放括号中。
■ 格式（严格遵守）：
  - 每条独占一行
  - 章：第X章 中文名 (English) (p.X - p.Y)
  - 节：X.X 中文名 (English) (p.X - p.Y)
  - 章与章之间空一行

示例：

第1章 数值序列 (Numerical Sequences) (p.5 - p.14)
  1.1 实数序列 (Sequences of Real Numbers) (p.5 - p.14)

第2章 数值级数 (Numerical Series) (p.15 - p.28)
  2.0.1 非负项级数 (Series with Non-negative Terms) (p.17 - p.17)

要求：每条标注页码，不省略，只输出目录。

文本：
{toc_text}"""

PROMPT_EX = """基于以下知识点，针对【{query}】出题。中文出题，专业名词标英文。

### 基础题
**题目**：...
**提示**：...
**答案**：...

### 思考题
**题目**：...
**提示**：...
**答案**：...

知识点：
{content}"""

# ================================================================
#  主流程
# ================================================================
uploaded = st.file_uploader("UPLOAD_PDF", type="pdf")

# 状态判断
has_api = bool(api_key)
has_file = uploaded is not None
is_done = st.session_state.processing_done

# 引导面板：未完成初始化时显示
if not is_done:
    step1_cls = "ctx-guide-done" if has_api else "ctx-guide-active"
    step2_cls = "ctx-guide-done" if has_file else ("ctx-guide-active" if has_api else "")
    step3_cls = "ctx-guide-active" if (has_api and has_file) else ""

    st.markdown(f"""
    <div class="ctx-guide">
        <div class="ctx-guide-title">初始化流程 // INIT_SEQUENCE</div>
        <div class="ctx-guide-step {step1_cls}">
            <span class="ctx-guide-num">01</span>
            {"✓ API 已连接" if has_api else "← 点击左上角 ❯❯ 打开侧边栏，输入 API Key"}
        </div>
        <div class="ctx-guide-step {step2_cls}">
            <span class="ctx-guide-num">02</span>
            {"✓ PDF 已上传" if has_file else "上传 PDF 文件"}
        </div>
        <div class="ctx-guide-step {step3_cls}">
            <span class="ctx-guide-num">03</span>
            {"就绪，正在处理..." if (has_api and has_file) else "等待上述步骤完成后自动开始"}
        </div>
    </div>
    """, unsafe_allow_html=True)

if uploaded and api_key:
    if not st.session_state.processing_done:
        with st.spinner("PROCESSING..."):
            try:
                db, pts, toc, n = process_pdf(uploaded, toc_pages, chunk_size)
                st.session_state.vector_db = db
                st.session_state.page_texts = pts
                llm = get_llm()
                r = llm.invoke(PROMPT_TOC.format(toc_text=toc))
                st.session_state.framework = r.content
                st.session_state.toc_entries = parse_toc_text(r.content)
                st.session_state.processing_done = True
                st.rerun()
            except Exception as e:
                st.error(f"ERR: {e}")

# ================================================================
#  目录 + 提取
# ================================================================
if st.session_state.framework and st.session_state.toc_entries:
    entries = st.session_state.toc_entries
    opts = build_opts(entries)

    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown("#### 📑 INDEX")
        st.markdown(render_toc_html(entries), unsafe_allow_html=True)

    with c2:
        st.markdown("#### 🔍 EXTRACT")

        if opts:
            dl = [o["display"] for o in opts]
            ci = st.selectbox("TARGET", range(len(dl)), format_func=lambda i: dl[i])
            chosen = opts[ci]
            ql = chosen["label"]; ps = chosen["ps"]; pe = chosen["pe"]
            st.caption(f"RANGE: p.{ps}–{pe}")
        else:
            ql = st.text_input("TARGET"); ps = pe = 0

        gen_ex = st.checkbox("GENERATE_EXERCISE", value=False)

        if st.button("▶ EXECUTE", use_container_width=True):
            if not ql: st.warning("SELECT TARGET")
            else:
                try:
                    llm = get_llm()
                    if ps > 0:
                        ctx = get_pages(ps, pe)
                        if not ctx.strip(): st.error("NO_CONTENT"); st.stop()
                        result = batch_extract(ctx, ql, llm, max_chars)
                    else:
                        ctx = sem_search(ql, retrieve_k)
                        result = extract_single(ctx, ql, llm)

                    if isinstance(result, list):
                        st.session_state.knowledge_points = result
                        st.session_state.raw_detail = json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        st.session_state.knowledge_points = None
                        st.session_state.raw_detail = result if isinstance(result,str) else str(result)
                    st.session_state.current_query = ql

                    if gen_ex:
                        with st.spinner("GENERATING..."):
                            c = json.dumps(result,ensure_ascii=False) if isinstance(result,list) else str(result)
                            st.session_state.exercises = llm.invoke(
                                PROMPT_EX.format(query=ql, content=c)).content
                    st.rerun()
                except Exception as e:
                    st.error(f"ERR: {e}")

# ================================================================
#  结果
# ================================================================
if st.session_state.knowledge_points:
    kps = st.session_state.knowledge_points
    ql = st.session_state.current_query

    st.markdown("---")
    st.markdown(f"#### 📊 {ql}")

    total = len(kps)
    hi = len([k for k in kps if k.get("weights",{}).get("core",0)>=4])
    ad = sum(k.get("weights",{}).get("difficulty",0) for k in kps)/max(total,1)
    mc = sum(1 for k in kps if st.session_state.mastered.get(k.get("id",""),False))
    st.markdown(f"""<div class="ctx-stats">
<div class="ctx-stat"><div class="ctx-stat-n">{total}</div><div class="ctx-stat-l">total</div></div>
<div class="ctx-stat"><div class="ctx-stat-n">{hi}</div><div class="ctx-stat-l">core</div></div>
<div class="ctx-stat"><div class="ctx-stat-n">{ad:.1f}</div><div class="ctx-stat-l">avg_diff</div></div>
<div class="ctx-stat"><div class="ctx-stat-n">{mc}/{total}</div><div class="ctx-stat-l">mastered</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="ctx-lgd">
<span><span class="ctx-lgd-d" style="background:#fff">　</span>S</span>
<span><span class="ctx-lgd-d" style="background:#888">　</span>A</span>
<span><span class="ctx-lgd-d" style="background:#555">　</span>B</span>
<span><span class="ctx-lgd-d" style="background:#333">　</span>C</span>
<span><span class="ctx-lgd-d" style="background:#1a1a1a">　</span>D</span>
</div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["TREEMAP","CARDS","MATRIX","RAW"])

    with t1:
        st.plotly_chart(make_treemap(kps), use_container_width=True)

    with t2:
        fc1,fc2,fc3 = st.columns(3)
        with fc1: fcore = st.select_slider("MIN_CORE",[1,2,3,4,5],value=1)
        with fc2: fsort = st.selectbox("SORT",["CORE","DIFF","EXAM","PAGE"])
        with fc3: fstat = st.selectbox("STATUS",["ALL","TODO","DONE"])

        fl = [k for k in kps if k.get("weights",{}).get("core",0)>=fcore]
        if fstat=="TODO": fl=[k for k in fl if not st.session_state.mastered.get(k.get("id",""),False)]
        elif fstat=="DONE": fl=[k for k in fl if st.session_state.mastered.get(k.get("id",""),False)]
        sm = {
            "CORE":lambda k:-k.get("weights",{}).get("core",0),
            "DIFF":lambda k:-k.get("weights",{}).get("difficulty",0),
            "EXAM":lambda k:-k.get("weights",{}).get("exam_weight",0),
            "PAGE":lambda k:int(re.search(r"\d+",k.get("page","0")).group()) if re.search(r"\d+",k.get("page","0")) else 0,
        }
        fl.sort(key=sm.get(fsort,sm["CORE"]))
        st.caption(f"SHOWING {len(fl)}/{len(kps)}")

        for i, kp in enumerate(fl):
            cc1,cc2 = st.columns([7,1])
            with cc1: st.markdown(card_html(kp), unsafe_allow_html=True)
            with cc2:
                kid = kp.get("id",f"k{i}")
                mv = st.checkbox("✓",value=st.session_state.mastered.get(kid,False),key=f"m_{kid}")
                st.session_state.mastered[kid] = mv

            with st.expander(f"DETAIL // {kp['name']}"):
                rc1,rc2 = st.columns([1,1])
                with rc1: st.plotly_chart(make_radar(kp), use_container_width=True)
                with rc2:
                    w = kp.get("weights",{}); tot = sum(w.values()); tn = tier(w.get("core",0))
                    st.markdown(f"**{tn} // {tot}/20**")
                    st.markdown(f"""
| DIM | VAL | NOTE |
|---|---|---|
| CORE | {w.get('core',0)}/5 | {"必须掌握" if w.get('core',0)>=4 else "建议了解" if w.get('core',0)>=3 else "可选"} |
| DIFF | {w.get('difficulty',0)}/5 | {"重点攻克" if w.get('difficulty',0)>=4 else "中等" if w.get('difficulty',0)>=3 else "容易"} |
| CONN | {w.get('connectivity',0)}/5 | {"枢纽" if w.get('connectivity',0)>=4 else "有关联" if w.get('connectivity',0)>=3 else "独立"} |
| EXAM | {w.get('exam_weight',0)}/5 | {"高频" if w.get('exam_weight',0)>=4 else "可能" if w.get('exam_weight',0)>=3 else "低频"} |
""")

    with t3:
        st.plotly_chart(make_bubble(kps), use_container_width=True)

    with t4:
        st.code(st.session_state.raw_detail, language="json")

    if st.session_state.exercises:
        st.markdown("---")
        st.markdown("#### ✏️ EXERCISE")
        st.markdown(st.session_state.exercises)

elif st.session_state.raw_detail and not st.session_state.knowledge_points:
    st.markdown("---")
    st.caption("PARSE_FAILED // TEXT_MODE")
    raw = st.session_state.raw_detail
    last = robust_json(raw)
    if last: st.session_state.knowledge_points = last; st.rerun()
    else: st.code(raw)
