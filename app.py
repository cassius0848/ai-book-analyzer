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
#  PRISM UI — 深黑终端风格
# ================================================================
# Check if API key is in secrets to determine sidebar state
_has_secret_key = False
try:
    _has_secret_key = bool(st.secrets.get("my_deepseek_key"))
except Exception:
    pass

st.set_page_config(
    page_title="PRISM // 文件解构",
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
.kp-intu {
    font-family: var(--sans); font-size: 11px;
    color: var(--dim); line-height: 1.6;
    margin: 0 0 8px 0;
    padding: 6px 10px;
    border-left: 1px solid var(--grid);
    font-style: italic;
}

/* 详情面板 */
.kp-detail-section {
    padding: 10px 0;
    border-bottom: 1px solid var(--grid);
}
.kp-detail-section:last-child { border-bottom: none; }
.kp-detail-label {
    font-family: var(--mono); font-size: 9px;
    color: var(--dim); text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 6px;
}
.kp-detail-body {
    font-family: var(--sans); font-size: 12px;
    color: var(--main); line-height: 1.8;
}
.kp-formula {
    font-family: var(--mono); font-size: 11px;
    color: var(--hi); padding: 8px 12px;
    background: var(--bg); border: 1px solid var(--grid);
    margin: 4px 0;
}
.kp-pitfall {
    font-family: var(--sans); font-size: 11px;
    color: #c0c0c0; padding: 6px 10px;
    border-left: 2px solid #555;
    margin: 4px 0;
}
</style>
""", unsafe_allow_html=True)

# 顶栏
st.markdown("""
<div class="ctx-header">
    <div>
        <div class="ctx-title">◇ PRISM 文件解构助手</div>
        <div class="ctx-subtitle">一本书的光谱分析 — 上传 · 解构 · 洞察 · 掌握</div>
    </div>
    <div>
        <span class="sys">SYS</span> <span class="val">PRISM_ENGINE</span>
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
    "synthesis": "", "deep_dives": {},
    "extract_ctx": "", "extract_ps": 0, "extract_pe": 0,
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
    st.markdown("#### PRISM_CONFIG")
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
#  提取引擎（两阶段深度解构）
# ================================================================

# 第一阶段：全量解构 — 提取所有知识点 + 完整教学内容
PHASE1_PROMPT = """你是一位严谨的教研专家。用户是初学者，想深度学习【{query}】。
【最高优先级】用户阅读你的输出后，不需要再翻阅原文。你的输出必须完整替代原文的教学功能。

以下是原文{batch_note}（方括号内为页码）。

■ 语言：整体中文，专业术语用「中文名 (English Term)」。
■ 权重（1-5）：core核心度, difficulty难度, connectivity关联度, exam_weight考试权重
■ 输出：只输出 JSON 数组，直接 [ 开头 ] 结尾，不要其他文字。

每个元素必须包含以下字段，每个字段都要尽可能详尽：

{{
  "id": "KP-01",
  "name": "稳定匹配 (Stable Matching)",
  "category": "所属主题",
  "page": "p.1-p.12",
  "weights": {{"core":4,"difficulty":3,"connectivity":5,"exam_weight":4}},
  "prerequisites": ["KP-XX"],
  "related": ["KP-XX"],

  "definition": "【必须完整】(1)概念本质 (2)形式化定义含LaTeX公式 (3)适用条件 (4)关键性质。不少于5句话。原文对此概念的所有描述都要整合进来。",

  "theorems": "【原文相关的全部定理/引理/推论】每条含：名称、完整陈述（LaTeX）、证明核心思路2-3句。无则写'无'。",

  "formulas": ["每一个公式含变体，LaTeX格式，不省略"],

  "intuition": "【通俗解释】(1)生活类比 (2)为什么需要这个概念 (3)和前置概念的关系。3-5句。",

  "examples": "【完整例题含解题过程】(1)题目 (2)逐步求解每步用LaTeX (3)答案。原文有多个例题全部提取，用---分隔。原文无例题则自行构造。",

  "pitfalls": "【2-3个易错点】含：错误描述→为什么错→正确理解。",

  "key_conclusions": "【1-3条核心结论】掌握后应记住的要点。"
}}

■ 必须遵守（违反则提取失败）：
1. 原文每一个定理/引理/推论 → 必须出现在某KP的 theorems
2. 原文每一个例题 → 必须出现在某KP的 examples（含完整解题）
3. 原文每一个公式 → 必须出现在某KP的 formulas
4. definition 详细到读完等同于读了原文
5. 图表用文字描述其传达的信息

原文：
{text}"""

# 第二阶段：章节综述 — 把零散知识点串成认知通路
SYNTHESIS_PROMPT = """你是一位优秀的认知科学导师。以下是从【{query}】中提取的所有知识点（JSON格式）。

你的目标不是简单总结，而是帮助初学者在脑中建立这些概念之间的「神经通路」——让每个概念自然地连接到下一个。

请生成一份完整的章节深度学习指南：

### 📍 定位
- 这一章在整个学科知识体系中的坐标位置（2-3句）
- 学完这一章后，你能解决什么问题？具体的能力增益是什么？

### 🧠 认知路径
用**连贯的段落**（不是列表），按照概念之间的逻辑递进关系，模拟一个初学者从零开始理解这些概念的思维过程。
- 从最基础的概念开始，逐步引入更复杂的概念
- 每引入一个新概念时，解释它和前面概念的关系（"因为我们已经知道了X，所以自然会问Y..."）
- 每个概念标注ID如[KP-01]
- 对于抽象概念，用**具体的类比或生活场景**帮助理解
- 对于有公式的概念，解释公式每个符号的含义和公式的直觉意义

### 📐 公式速查表
集中列出所有重要公式/定理，每个标注来源[KP-XX]，并用一句话说明"什么时候用这个公式"

### ⚡ 关键连接
列出3-5组最重要的概念关联对（如"A→B：理解A是理解B的前提，因为..."），帮助加固神经通路

### ⚠️ 认知陷阱
3-5条最容易出错的思维误区（不只是计算错误，更关注概念理解上的偏差）

### 🎯 掌握度自测
5个快速判断题（对/错/需补充），覆盖核心概念。每题给出答案和1句解释。

■ 语言：中文为主，专业术语「中文 (English)」格式
■ 公式用 $LaTeX$ 格式
■ 写法要像在和学生对话，不要像论文

知识点数据：
{kps_json}"""

def extract_single(text, query, llm, bl=None):
    bn = f"（第{bl}部分）" if bl else ""
    resp = llm.invoke(PHASE1_PROMPT.format(query=query, batch_note=bn, text=text)).content
    p = robust_json(resp)
    return p if p else resp

def batch_extract(ft, query, llm, mc):
    if len(ft) <= mc:
        return extract_single(ft, query, llm)
    pages = ft.split("\n\n")
    batches, cur = [], ""
    for b in pages:
        if len(cur)+len(b)>mc and cur: batches.append(cur); cur=b
        else: cur += ("\n\n"+b) if cur else b
    if cur: batches.append(cur)
    all_kps = []
    prog = st.progress(0, text="REFRACTING...")
    for i, batch in enumerate(batches):
        prog.progress((i+1)/(len(batches)+1), text=f"LAYER {i+1}/{len(batches)}")
        r = extract_single(batch, query, llm, f"{i+1}/{len(batches)}")
        if isinstance(r, list): all_kps.extend(r)
    prog.progress(1.0, text="CONVERGING...")
    if not all_kps: prog.empty(); return "NULL_SPECTRUM"
    rp = f"""合并以下知识点去重。只输出 JSON 数组，直接 [ 开头 ] 结尾。
保持所有字段完整（definition, theorems, formulas, intuition, examples, pitfalls, key_conclusions），合并时选择信息量最大的版本，不要丢弃任何内容。
保持中文，专业名词「中文 (English)」。
{json.dumps(all_kps, ensure_ascii=False)}"""
    resp = llm.invoke(rp).content; prog.empty()
    p = robust_json(resp)
    return p if p else all_kps

def generate_synthesis(kps, query, llm):
    """生成章节综述"""
    kps_json = json.dumps(kps, ensure_ascii=False, indent=1)
    resp = llm.invoke(SYNTHESIS_PROMPT.format(query=query, kps_json=kps_json)).content
    return resp

# ================================================================
#  可视化（PRISM 配色）
# ================================================================
def tier(c):
    if c>=5: return "S"
    if c>=4: return "A"
    if c>=3: return "B"
    if c>=2: return "C"
    return "D"

def card_html(kp):
    w = kp.get("weights",{}); c = w.get("core",3); t = tier(c)
    # 在卡片中显示 definition + intuition 预览
    defn = kp.get('definition','—')
    intuition = kp.get('intuition','')
    intuition_html = f'<div class="kp-intu">{intuition}</div>' if intuition else ''

    return f"""<div class="kp-card t-{t}">
<div class="kp-head"><span class="kp-tier">{t}</span><span class="kp-name">{kp.get('id','')}  {kp.get('name','')}</span></div>
<div class="kp-def">{defn}</div>
{intuition_html}
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
    root = "PRISM"
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
#  神经通路引擎 (Neural Pathway Engine)
# ================================================================

def build_graph(kps):
    """构建知识点依赖图：节点=知识点，边=prerequisites+related"""
    id_map = {kp.get("id",""):kp for kp in kps}
    edges_pre = []   # 强连接（前置依赖）
    edges_rel = []   # 弱连接（相关联系）
    for kp in kps:
        kid = kp.get("id","")
        for pre in kp.get("prerequisites",[]):
            if pre in id_map:
                edges_pre.append((pre, kid))
        for rel in kp.get("related",[]):
            if rel in id_map and rel != kid:
                edges_rel.append((kid, rel))
    return id_map, edges_pre, edges_rel


def compute_learning_path(kps, mastered_set):
    """
    基于拓扑排序 + 权重优先级计算最优学习路径。
    优先学：(1)前置依赖已满足 (2)核心度高 (3)难度低（先易后难）
    跳过已掌握的节点。
    """
    id_map = {kp.get("id",""):kp for kp in kps}
    # 计算入度
    in_deg = {kp.get("id",""):0 for kp in kps}
    children = {kp.get("id",""):[] for kp in kps}
    for kp in kps:
        kid = kp.get("id","")
        for pre in kp.get("prerequisites",[]):
            if pre in id_map:
                in_deg[kid] = in_deg.get(kid,0) + 1
                children.setdefault(pre,[]).append(kid)

    # 拓扑排序 + 权重优先队列
    import heapq
    # 可用节点：入度为0 or 前置都已掌握
    available = []
    for kid, deg in in_deg.items():
        if deg == 0 or all(p in mastered_set for p in id_map.get(kid,{}).get("prerequisites",[])):
            kp = id_map[kid]
            w = kp.get("weights",{})
            # 排序: 核心度高优先(负), 难度低优先(正)
            priority = (-w.get("core",3), w.get("difficulty",3))
            heapq.heappush(available, (priority, kid))

    path = []
    visited = set(mastered_set)
    while available:
        _, kid = heapq.heappop(available)
        if kid in visited:
            continue
        visited.add(kid)
        path.append(id_map[kid])
        # 解锁下游
        for child in children.get(kid,[]):
            # 检查 child 的所有前置是否已满足
            prereqs = id_map.get(child,{}).get("prerequisites",[])
            if all(p in visited for p in prereqs if p in id_map):
                w = id_map[child].get("weights",{})
                heapq.heappush(available, ((-w.get("core",3), w.get("difficulty",3)), child))

    # 补充没有被拓扑排序覆盖的孤立节点
    for kp in kps:
        if kp.get("id","") not in visited and kp.get("id","") not in mastered_set:
            path.append(kp)

    return path


def get_activation_map(kps, mastered_set):
    """计算激活状态：已掌握/可解锁/被阻塞"""
    id_map = {kp.get("id",""):kp for kp in kps}
    status = {}
    for kp in kps:
        kid = kp.get("id","")
        if kid in mastered_set:
            status[kid] = "active"     # 已激活
        else:
            prereqs = [p for p in kp.get("prerequisites",[]) if p in id_map]
            if not prereqs or all(p in mastered_set for p in prereqs):
                status[kid] = "ready"  # 可解锁
            else:
                status[kid] = "blocked" # 被阻塞
    return status


def make_network(kps, mastered_set):
    """生成知识网络图（神经通路可视化）"""
    id_map, edges_pre, edges_rel = build_graph(kps)
    activation = get_activation_map(kps, mastered_set)

    # 简单力导向布局（圆形 + 扰动）
    import math
    n = len(kps)
    positions = {}
    for i, kp in enumerate(kps):
        angle = 2 * math.pi * i / max(n, 1)
        # 核心度高的节点靠内
        w = kp.get("weights",{})
        radius = 1.5 - (w.get("core",3) / 5) * 0.8
        positions[kp.get("id","")] = (radius * math.cos(angle), radius * math.sin(angle))

    fig = go.Figure()

    # 绘制弱连接（灰色细线）
    for src, tgt in edges_rel:
        if src in positions and tgt in positions:
            x0, y0 = positions[src]; x1, y1 = positions[tgt]
            fig.add_trace(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=0.5, color="#222"), hoverinfo="none", showlegend=False))

    # 绘制强连接（白色线 + 箭头效果）
    for src, tgt in edges_pre:
        if src in positions and tgt in positions:
            x0, y0 = positions[src]; x1, y1 = positions[tgt]
            fig.add_trace(go.Scatter(
                x=[x0,x1,None], y=[y0,y1,None], mode="lines",
                line=dict(width=1.5, color="#555"), hoverinfo="none", showlegend=False))

    # 绘制节点
    color_map = {"active":"#fff", "ready":"#888", "blocked":"#333"}
    size_map = {"active":18, "ready":14, "blocked":10}
    border_map = {"active":"#fff", "ready":"#666", "blocked":"#222"}

    for kp in kps:
        kid = kp.get("id","")
        if kid not in positions: continue
        x, y = positions[kid]
        st_status = activation.get(kid, "blocked")
        w = kp.get("weights",{})
        sz = size_map[st_status] + w.get("core",3) * 2

        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers+text",
            marker=dict(size=sz, color=color_map[st_status],
                       line=dict(width=1.5, color=border_map[st_status])),
            text=kp.get("id",""), textposition="top center",
            textfont=dict(size=8, color="#666", family="JetBrains Mono"),
            hovertext=f"{kp.get('name','')}<br>状态: {'已激活' if st_status=='active' else '可解锁' if st_status=='ready' else '被阻塞'}<br>核心度: {w.get('core',0)}/5",
            hoverinfo="text", showlegend=False))

    fig.update_layout(
        paper_bgcolor="#030303", plot_bgcolor="#030303",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2,2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2,2],
                   scaleanchor="x", scaleratio=1),
        height=450, margin=dict(t=10,b=10,l=10,r=10),
        font=dict(family="JetBrains Mono"))
    return fig


def render_learning_path(path, mastered_set):
    """渲染学习路径为 HTML"""
    html = ""
    for i, kp in enumerate(path):
        kid = kp.get("id","")
        w = kp.get("weights",{})
        is_done = kid in mastered_set
        t = tier(w.get("core",3))
        opacity = "0.4" if is_done else "1"
        strike = "text-decoration:line-through;" if is_done else ""
        status_icon = "✓" if is_done else f"{i+1:02d}"
        signal = w.get("core",3) * 20  # 信号强度百分比

        html += f"""<div style="display:flex;align-items:center;gap:10px;padding:8px 12px;
            border-bottom:1px solid var(--grid);opacity:{opacity};font-family:var(--mono);">
            <span style="color:var(--dim);min-width:24px;font-size:10px;">{status_icon}</span>
            <span style="min-width:18px;font-size:9px;font-weight:700;
                border:1px solid var(--dim);padding:1px 4px;color:var(--hi);">{t}</span>
            <span style="flex:1;font-family:var(--sans);font-size:11px;color:var(--hi);{strike}">{kp.get('name','')}</span>
            <div style="width:60px;height:4px;background:var(--grid);position:relative;">
                <div style="width:{signal}%;height:100%;background:var(--hi);"></div>
            </div>
            <span style="color:var(--dim);font-size:9px;min-width:30px;">{kp.get('page','')}</span>
        </div>"""
    return html

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
uploaded = st.file_uploader("LOAD_SOURCE", type="pdf")

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
        <div class="ctx-guide-title">初始化 // PRISM_INIT</div>
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
        with st.spinner("REFRACTING..."):
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
        st.markdown("#### 📑 SPECTRUM")
        st.markdown(render_toc_html(entries), unsafe_allow_html=True)

    with c2:
        st.markdown("#### ◇ REFRACT")

        if opts:
            dl = [o["display"] for o in opts]
            ci = st.selectbox("TARGET", range(len(dl)), format_func=lambda i: dl[i])
            chosen = opts[ci]
            ql = chosen["label"]; ps = chosen["ps"]; pe = chosen["pe"]
            st.caption(f"RANGE: p.{ps}–{pe}")
        else:
            ql = st.text_input("TARGET"); ps = pe = 0

        gen_ex = st.checkbox("GENERATE_EXERCISE", value=False)

        if st.button("◇ REFRACT", use_container_width=True):
            if not ql: st.warning("SELECT_TARGET")
            else:
                try:
                    llm = get_llm()
                    if ps > 0:
                        ctx = get_pages(ps, pe)
                        if not ctx.strip(): st.error("EMPTY_RANGE"); st.stop()
                        result = batch_extract(ctx, ql, llm, max_chars)
                        st.session_state["extract_ctx"] = ctx  # 保存原文用于深度展开
                        st.session_state["extract_ps"] = ps
                        st.session_state["extract_pe"] = pe
                    else:
                        ctx = sem_search(ql, retrieve_k)
                        result = extract_single(ctx, ql, llm)
                        st.session_state["extract_ctx"] = ctx
                        st.session_state["extract_ps"] = 0
                        st.session_state["extract_pe"] = 0

                    if isinstance(result, list):
                        st.session_state.knowledge_points = result
                        st.session_state.raw_detail = json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        st.session_state.knowledge_points = None
                        st.session_state.raw_detail = result if isinstance(result,str) else str(result)
                    st.session_state.current_query = ql
                    st.session_state["synthesis"] = ""
                    st.session_state["deep_dives"] = {}  # per-KP deep content cache

                    # ── 自动生成综述（不再需要按钮）──
                    if isinstance(result, list) and result:
                        with st.spinner("SYNTHESIZING..."):
                            st.session_state["synthesis"] = generate_synthesis(result, ql, llm)

                    if gen_ex:
                        with st.spinner("COMPOSING..."):
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
    st.markdown(f"#### ◇ {ql}")

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

    t0,t1,t2,t5,t6,t3,t4 = st.tabs(["OVERVIEW","SPECTRUM","LAYERS","NETWORK","PATHWAY","MATRIX","RAW"])

    # ── OVERVIEW: 自动生成的章节深度指南 ──
    with t0:
        if st.session_state.get("synthesis"):
            st.markdown(st.session_state["synthesis"])
        else:
            st.caption("综述生成中或数据不足，请稍候...")

    # ── SPECTRUM: Treemap ──
    with t1:
        st.plotly_chart(make_treemap(kps), use_container_width=True)

    # ── LAYERS: 知识卡片 + 深度详情 ──
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
        st.caption(f"VISIBLE {len(fl)}/{len(kps)}")

        for i, kp in enumerate(fl):
            cc1,cc2 = st.columns([7,1])
            with cc1: st.markdown(card_html(kp), unsafe_allow_html=True)
            with cc2:
                kid = kp.get("id",f"k{i}")
                mv = st.checkbox("✓",value=st.session_state.mastered.get(kid,False),key=f"m_{kid}")
                st.session_state.mastered[kid] = mv

            with st.expander(f"◇ {kp['name']} — 完整学习内容"):
                # 权重雷达 + 权重表
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

                st.markdown("---")

                # 📐 定理/引理
                theorems = kp.get("theorems", "")
                if theorems and theorems != "无":
                    st.markdown("**THEOREMS / LEMMAS**")
                    st.markdown(theorems)

                # 📐 公式
                formulas = kp.get("formulas", [])
                if formulas and any(f.strip() for f in formulas if isinstance(f, str)):
                    st.markdown("**FORMULAS**")
                    for f in formulas:
                        if isinstance(f, str) and f.strip():
                            st.latex(f.replace("$","").strip())

                # 💡 直觉
                intuition = kp.get("intuition", "")
                if intuition:
                    st.markdown("**INTUITION**")
                    st.info(intuition)

                # 📖 例题（支持多个）
                examples = kp.get("examples", kp.get("example", ""))
                if examples:
                    st.markdown("**EXAMPLES**")
                    if isinstance(examples, str):
                        for ex in examples.split("---"):
                            if ex.strip():
                                st.markdown(ex.strip())
                                st.markdown("")
                    elif isinstance(examples, list):
                        for ex in examples:
                            st.markdown(ex)
                            st.markdown("")

                # ⚠️ 易错点
                pitfalls = kp.get("pitfalls", "")
                if pitfalls:
                    st.markdown("**PITFALLS**")
                    st.warning(pitfalls)

                # 🎯 核心结论
                conclusions = kp.get("key_conclusions", "")
                if conclusions:
                    st.markdown("**KEY_CONCLUSIONS**")
                    st.success(conclusions)

                # 前置/相关
                pre = kp.get("prerequisites",[])
                rel = kp.get("related",[])
                if pre: st.caption(f"PREREQ: {', '.join(pre)}")
                if rel: st.caption(f"RELATED: {', '.join(rel)}")

                # ── 深度展开：用原文重新生成完整讲解 ──
                deep_key = f"deep_{kp.get('id','')}"
                if st.session_state.get("deep_dives",{}).get(deep_key):
                    st.markdown("---")
                    st.markdown("**DEEP_DIVE // 完整讲解**")
                    st.markdown(st.session_state["deep_dives"][deep_key])
                else:
                    if st.button(f"◇ DEEP_DIVE — 生成完整讲解", key=f"dd_{kp.get('id','')}"):
                        with st.spinner("DEEP_DIVING..."):
                            try:
                                llm = get_llm()
                                # 取该知识点对应的原文页
                                page_str = kp.get("page","")
                                pm = re.findall(r"\d+", page_str)
                                if pm:
                                    dps = int(pm[0])
                                    dpe = int(pm[-1]) if len(pm)>1 else dps
                                    deep_ctx = get_pages(dps, dpe)
                                else:
                                    deep_ctx = st.session_state.get("extract_ctx","")

                                deep_prompt = f"""你是一位认知科学导师。请针对【{kp.get('name','')}】生成一份完整的专题讲解。
用户读完后应完全掌握这个概念，不需要看原文。

要求：
1. 从最基础的理解开始，逐步深入
2. 所有公式都要解释每个符号的含义
3. 所有定理都要解释证明思路（不需要完整证明，但要说清为什么成立）
4. 至少包含2个具体的数值例题（含完整解题过程）
5. 指出3个常见误区
6. 用生活类比帮助理解抽象概念
7. 最后给出3个自测问题（附答案）

语言：中文，专业名词「中文 (English)」，公式用 $LaTeX$

参考原文：
{deep_ctx}

知识点概要：
{json.dumps(kp, ensure_ascii=False)}"""
                                deep_resp = llm.invoke(deep_prompt).content
                                if "deep_dives" not in st.session_state:
                                    st.session_state["deep_dives"] = {}
                                st.session_state["deep_dives"][deep_key] = deep_resp
                                st.rerun()
                            except Exception as e:
                                st.error(f"ERR: {e}")

    # ── NETWORK: 知识神经网络 ──
    with t5:
        mastered_set = {k for k,v in st.session_state.mastered.items() if v}
        st.plotly_chart(make_network(kps, mastered_set), use_container_width=True)

        # 激活统计
        act = get_activation_map(kps, mastered_set)
        n_active = sum(1 for v in act.values() if v=="active")
        n_ready = sum(1 for v in act.values() if v=="ready")
        n_blocked = sum(1 for v in act.values() if v=="blocked")

        st.markdown(f"""<div style="display:flex;gap:24px;padding:8px 0;font-family:var(--mono);font-size:10px;">
<span style="color:#fff;">● ACTIVE {n_active}</span>
<span style="color:#888;">● READY {n_ready}</span>
<span style="color:#333;">● BLOCKED {n_blocked}</span>
<span style="color:var(--dim);">—— 粗线=前置依赖 · 细线=相关联系 · 节点大小=核心度</span>
</div>""", unsafe_allow_html=True)

        st.caption("勾选 LAYERS 中的 ✓ 标记已掌握 → 网络自动更新激活状态")

    # ── PATHWAY: 最优学习路径 ──
    with t6:
        mastered_set = {k for k,v in st.session_state.mastered.items() if v}
        path = compute_learning_path(kps, mastered_set)

        remaining = [kp for kp in path if kp.get("id","") not in mastered_set]
        st.markdown(f"""<div style="font-family:var(--mono);font-size:10px;color:var(--dim);
            padding:8px 0;border-bottom:1px solid var(--grid);text-transform:uppercase;
            display:flex;justify-content:space-between;">
<span>OPTIMAL LEARNING SEQUENCE</span>
<span>REMAINING: {len(remaining)} / {len(kps)}</span>
</div>""", unsafe_allow_html=True)

        st.markdown(render_learning_path(path, mastered_set), unsafe_allow_html=True)

        st.caption("路径逻辑：拓扑排序（前置依赖优先）→ 核心度高优先 → 先易后难 · 信号条 = 核心度")

    # ── MATRIX: 气泡图 ──
    with t3:
        st.plotly_chart(make_bubble(kps), use_container_width=True)

    # ── RAW ──
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
