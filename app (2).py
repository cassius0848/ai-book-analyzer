import re
import json
import streamlit as st
import fitz  # PyMuPDF
import plotly.graph_objects as go
import plotly.express as px
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ================================================================
#                        页面 & 样式
# ================================================================
st.set_page_config(page_title="AI 知识提取器", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }

/* ---------- 知识卡片 ---------- */
.kp-card {
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border-left: 6px solid;
    background: #fff;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    transition: all 0.25s ease;
}
.kp-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.12);
}
.kp-card.tier-S { border-left-color: #e74c3c; background: linear-gradient(135deg, #fef2f2 0%, #fff 100%); }
.kp-card.tier-A { border-left-color: #f39c12; background: linear-gradient(135deg, #fffbeb 0%, #fff 100%); }
.kp-card.tier-B { border-left-color: #3498db; background: linear-gradient(135deg, #eff6ff 0%, #fff 100%); }
.kp-card.tier-C { border-left-color: #2ecc71; background: linear-gradient(135deg, #f0fdf4 0%, #fff 100%); }
.kp-card.tier-D { border-left-color: #95a5a6; background: linear-gradient(135deg, #f9fafb 0%, #fff 100%); }

.kp-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 8px;
}
.kp-tier {
    font-size: 0.7em; font-weight: 900; padding: 2px 8px;
    border-radius: 6px; color: #fff; letter-spacing: 1px;
}
.tier-S .kp-tier { background: #e74c3c; }
.tier-A .kp-tier { background: #f39c12; }
.tier-B .kp-tier { background: #3498db; }
.tier-C .kp-tier { background: #2ecc71; }
.tier-D .kp-tier { background: #95a5a6; }

.kp-name {
    font-size: 1.1em; font-weight: 700; color: #1a1a2e;
}
.kp-def {
    font-size: 0.9em; color: #4b5563; line-height: 1.7;
    margin: 8px 0 12px 0; padding-left: 4px;
}
.kp-meta {
    display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
}
.kp-tag {
    font-size: 0.72em; padding: 3px 10px; border-radius: 20px;
    font-weight: 600; color: #fff; display: inline-flex; align-items: center; gap: 4px;
}
.tag-core { background: #e74c3c; }
.tag-diff { background: #8e44ad; }
.tag-conn { background: #2980b9; }
.tag-exam { background: #27ae60; }
.tag-page { background: #6b7280; }

/* ---------- 统计面板 ---------- */
.stat-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 16px 0;
}
.stat-card {
    text-align: center; padding: 18px 12px; border-radius: 14px; color: #fff;
}
.stat-card.s1 { background: linear-gradient(135deg, #667eea, #764ba2); }
.stat-card.s2 { background: linear-gradient(135deg, #e74c3c, #c0392b); }
.stat-card.s3 { background: linear-gradient(135deg, #f39c12, #e67e22); }
.stat-card.s4 { background: linear-gradient(135deg, #2ecc71, #27ae60); }
.stat-num { font-size: 2.2em; font-weight: 900; line-height: 1.2; }
.stat-label { font-size: 0.82em; opacity: 0.9; margin-top: 4px; }

/* ---------- 图例 ---------- */
.legend-bar {
    display: flex; gap: 16px; flex-wrap: wrap; padding: 8px 0; margin-bottom: 8px;
}
.legend-item { display: flex; align-items: center; gap: 5px; font-size: 0.8em; color: #666; }
.legend-dot { width: 14px; height: 14px; border-radius: 4px; }

/* ---------- TOC 目录 ---------- */
.toc-container {
    font-size: 0.88em; line-height: 2.0;
}
.toc-container .ch {
    font-weight: 700; font-size: 1.05em; color: #1a1a2e;
    margin-top: 12px; padding: 4px 0;
    border-bottom: 1px solid #e5e7eb;
}
.toc-container .sec {
    padding-left: 20px; color: #4b5563;
}
</style>
""", unsafe_allow_html=True)

st.title("📖 知识点全量提取助手")
st.caption("分而治之 · 权重思考模型 · 可视化知识图谱")

# ================================================================
#                      Session State
# ================================================================
_defaults = {
    "vector_db": None,
    "page_texts": {},
    "framework": "",
    "llm": None,
    "processing_done": False,
    "knowledge_points": None,
    "raw_detail": "",
    "mastered": {},
    "exercises": "",
    "current_query": "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ================================================================
#                        侧边栏配置
# ================================================================
with st.sidebar:
    st.header("⚙️ 配置中心")
    try:
        api_key = st.secrets["my_deepseek_key"]
        st.success("✅ API 已就绪")
    except Exception:
        api_key = st.text_input("输入 DeepSeek API Key", type="password")

    base_url = st.text_input("API 代理地址", value="https://api.deepseek.com")
    model_name = st.selectbox("选择模型", ["deepseek-chat", "deepseek-reasoner"])

    st.divider()
    st.header("📐 高级参数")
    toc_pages = st.slider("目录扫描页数", 5, 30, 15)
    chunk_size = st.slider("文本块大小", 500, 2000, 1000, step=100)
    retrieve_k = st.slider("语义检索块数 (k)", 4, 30, 15)
    max_chars_per_batch = st.slider("单批最大字符", 4000, 30000, 12000, step=2000)

# ================================================================
#                        核心函数
# ================================================================

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    if st.session_state.llm is None:
        st.session_state.llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    return st.session_state.llm

def process_pdf(file, toc_n, chunk_sz):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    toc_text, page_texts = "", {}
    all_chunks, all_metas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_sz, chunk_overlap=150)
    emb = load_embedding_model()
    for i, page in enumerate(doc):
        txt = page.get_text()
        pn = i + 1
        page_texts[pn] = txt
        if i < toc_n:
            toc_text += f"\n--- 第{pn}页 ---\n{txt}"
        for c in splitter.split_text(txt):
            all_chunks.append(c)
            all_metas.append({"page": pn})
    vdb = FAISS.from_texts(all_chunks, emb, metadatas=all_metas)
    return vdb, page_texts, toc_text, len(doc)

# ---------- 检索 ----------

def get_pages(ps, pe):
    pt = st.session_state.page_texts
    return "\n\n".join([f"[第{p}页]\n{pt[p]}" for p in range(ps, pe+1) if p in pt])

def sem_search(q, k):
    docs = st.session_state.vector_db.similarity_search(q, k=k)
    docs.sort(key=lambda d: d.metadata.get("page", 0))
    return "\n".join([f"[p.{d.metadata.get('page','?')}] {d.page_content}" for d in docs])

def parse_pages(text):
    m = re.findall(r"p\.?\s*(\d+)", text)
    if len(m) >= 2: return int(m[0]), int(m[1])
    if len(m) == 1: return int(m[0]), int(m[0]) + 20
    m = re.findall(r"第\s*(\d+)\s*页", text)
    if len(m) >= 2: return int(m[0]), int(m[1])
    if len(m) == 1: return int(m[0]), int(m[0]) + 20
    return 0, 0

# ================================================================
#               JSON 解析 (强化容错)
# ================================================================

def robust_parse_json(text):
    """
    多层容错解析 AI 返回的 JSON：
    1. 直接解析
    2. 去掉 markdown 标记后解析
    3. 逐行修复常见错误后解析
    4. 全部失败则返回 None
    """
    # 第一层：直接尝试
    try:
        data = json.loads(text)
        if isinstance(data, list) and len(data) > 0:
            return data
    except:
        pass

    # 第二层：清理 markdown 包裹
    cleaned = text
    cleaned = re.sub(r"```json\s*\n?", "", cleaned)
    cleaned = re.sub(r"```\s*\n?", "", cleaned)
    cleaned = cleaned.strip()

    # 找到最外层的 [ ... ]
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end+1]
        try:
            data = json.loads(candidate)
            if isinstance(data, list) and len(data) > 0:
                return data
        except:
            pass

        # 第三层：尝试修复常见问题
        fixed = candidate
        # 修复尾部多余逗号
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        # 修复单引号
        fixed = fixed.replace("'", '"')
        try:
            data = json.loads(fixed)
            if isinstance(data, list) and len(data) > 0:
                return data
        except:
            pass

    # 第四层：尝试逐个提取 JSON 对象
    objects = []
    for match in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            obj = json.loads(match.group())
            if "name" in obj:  # 至少有 name 字段
                objects.append(obj)
        except:
            continue
    if objects:
        return objects

    return None

# ================================================================
#               提取 & Map-Reduce
# ================================================================

EXTRACT_PROMPT = """你是一位严谨的教研专家。用户是初学者，想深度学习【{query}】。
以下是该章节的原文{batch_note}（方括号内为页码）。

📋 核心任务：逐页扫描，提取所有重要知识点，并评估权重。

📊 权重维度（1-5 分）：
• core（核心度）：5=绝对核心，1=补充细节
• difficulty（难度）：5=非常抽象，1=直观易懂
• connectivity（关联度）：5=关联多个概念，1=较独立
• exam_weight（考试权重）：5=必考，1=了解即可

⚠️ 输出要求：
1. 只输出一个 JSON 数组，不要输出任何其他文字
2. 不要用 markdown 代码块包裹
3. 不要输出解释、前言、总结
4. 直接以 [ 开头，以 ] 结尾

每个元素格式：
{{"id":"KP-01","name":"知识点名称","category":"所属主题","definition":"精准定义2-4句","page":"p.23","weights":{{"core":4,"difficulty":3,"connectivity":5,"exam_weight":4}},"prerequisites":["KP-XX"],"related":["KP-XX"]}}

原文：
{text}"""


def extract_single(text, query, llm, batch_label=None):
    bn = f"（第{batch_label}部分）" if batch_label else ""
    prompt = EXTRACT_PROMPT.format(query=query, batch_note=bn, text=text)
    resp = llm.invoke(prompt).content
    parsed = robust_parse_json(resp)
    if parsed is not None:
        return parsed
    # 解析失败返回原始文本
    return resp


def batch_extract(full_text, query, llm, max_chars):
    if len(full_text) <= max_chars:
        return extract_single(full_text, query, llm)

    # 按页分批
    pages = full_text.split("\n\n")
    batches, cur = [], ""
    for block in pages:
        if len(cur) + len(block) > max_chars and cur:
            batches.append(cur)
            cur = block
        else:
            cur += ("\n\n" + block) if cur else block
    if cur:
        batches.append(cur)

    all_kps = []
    progress = st.progress(0, text="📦 分批提取中...")
    for i, batch in enumerate(batches):
        progress.progress((i+1) / (len(batches)+1), text=f"🔍 第 {i+1}/{len(batches)} 批...")
        result = extract_single(batch, query, llm, batch_label=f"{i+1}/{len(batches)}")
        if isinstance(result, list):
            all_kps.extend(result)
        elif isinstance(result, str):
            # 某批解析失败，跳过
            continue

    progress.progress(1.0, text="🔄 合并去重中...")

    if not all_kps:
        progress.empty()
        return "提取失败，AI 未返回有效的结构化数据。"

    # Reduce 合并
    reduce_prompt = f"""你是教研专家。请合并以下知识点数据，去重后输出最终版本。

⚠️ 输出要求：只输出 JSON 数组，不要任何其他文字，直接以 [ 开头 ] 结尾。

数据：
{json.dumps(all_kps, ensure_ascii=False)}"""

    resp = llm.invoke(reduce_prompt).content
    progress.empty()
    parsed = robust_parse_json(resp)
    return parsed if parsed else all_kps  # 合并失败就用原始拼接


# ================================================================
#                     可视化组件
# ================================================================

def get_tier(core_score):
    if core_score >= 5: return "S", "🔴"
    if core_score >= 4: return "A", "🟠"
    if core_score >= 3: return "B", "🔵"
    if core_score >= 2: return "C", "🟢"
    return "D", "⚪"

def render_stat_panel(kps):
    total = len(kps)
    core_high = len([k for k in kps if k.get("weights",{}).get("core",0) >= 4])
    avg_diff = sum(k.get("weights",{}).get("difficulty",0) for k in kps) / max(total, 1)
    mastered = sum(1 for k in kps if st.session_state.mastered.get(k.get("id",""), False))
    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card s1"><div class="stat-num">{total}</div><div class="stat-label">📚 知识点总数</div></div>
        <div class="stat-card s2"><div class="stat-num">{core_high}</div><div class="stat-label">🔥 核心知识点</div></div>
        <div class="stat-card s3"><div class="stat-num">{avg_diff:.1f}</div><div class="stat-label">📊 平均难度</div></div>
        <div class="stat-card s4"><div class="stat-num">{mastered}/{total}</div><div class="stat-label">✅ 已掌握</div></div>
    </div>
    """, unsafe_allow_html=True)

def render_legend():
    st.markdown("""
    <div class="legend-bar">
        <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div> S 必须掌握</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div> A 重要</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div> B 一般</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div> C 了解</div>
        <div class="legend-item"><div class="legend-dot" style="background:#95a5a6"></div> D 补充</div>
    </div>
    """, unsafe_allow_html=True)

def render_card_html(kp):
    w = kp.get("weights", {})
    core = w.get("core", 3)
    tier, emoji = get_tier(core)
    tier_class = f"tier-{tier}"

    return f"""
    <div class="kp-card {tier_class}">
        <div class="kp-header">
            <span class="kp-tier">{emoji} {tier}级</span>
            <span class="kp-name">{kp.get('id','')} {kp.get('name','')}</span>
        </div>
        <div class="kp-def">{kp.get('definition','暂无定义')}</div>
        <div class="kp-meta">
            <span class="kp-tag tag-core">🎯 核心 {core}/5</span>
            <span class="kp-tag tag-diff">🔥 难度 {w.get('difficulty','?')}/5</span>
            <span class="kp-tag tag-conn">🔗 关联 {w.get('connectivity','?')}/5</span>
            <span class="kp-tag tag-exam">📝 考试 {w.get('exam_weight','?')}/5</span>
            <span class="kp-tag tag-page">📄 {kp.get('page','')}</span>
        </div>
    </div>"""

def render_treemap(kps):
    names, parents, values, colors, hovers = [], [], [], [], []
    cats = {}
    for kp in kps:
        cats.setdefault(kp.get("category", "未分类"), []).append(kp)

    root = "📖 知识图谱"
    names.append(root); parents.append(""); values.append(0); colors.append(0); hovers.append("")

    for cat, items in cats.items():
        names.append(f"📂 {cat}")
        parents.append(root); values.append(0); colors.append(0)
        hovers.append(f"<b>{cat}</b><br>包含 {len(items)} 个知识点")
        for kp in items:
            w = kp.get("weights", {})
            total = sum(w.get(k, 3) for k in ["core","difficulty","connectivity","exam_weight"])
            tier, emoji = get_tier(w.get("core", 3))
            names.append(f"{emoji} {kp['name']}")
            parents.append(f"📂 {cat}")
            values.append(total)
            colors.append(w.get("core", 3))
            hovers.append(
                f"<b>{kp['name']}</b><br>"
                f"🎯 核心度: {w.get('core',0)}/5<br>"
                f"🔥 难度: {w.get('difficulty',0)}/5<br>"
                f"🔗 关联度: {w.get('connectivity',0)}/5<br>"
                f"📝 考试: {w.get('exam_weight',0)}/5<br>"
                f"📄 {kp.get('page','?')}")

    fig = go.Figure(go.Treemap(
        labels=names, parents=parents, values=values,
        marker=dict(
            colors=colors,
            colorscale=[[0,"#dbeafe"],[0.3,"#93c5fd"],[0.5,"#fbbf24"],[0.7,"#f97316"],[1,"#dc2626"]],
            line=dict(width=2, color="white")),
        hovertext=hovers, hoverinfo="text", textinfo="label",
        textfont=dict(size=13), pathbar=dict(visible=True)))
    fig.update_layout(margin=dict(t=30,l=10,r=10,b=10), height=480,
                      font=dict(family="Noto Sans SC"))
    return fig

def render_radar(kp):
    w = kp.get("weights", {})
    cats = ["🎯 核心度","🔥 难度","🔗 关联度","📝 考试"]
    vals = [w.get("core",0), w.get("difficulty",0), w.get("connectivity",0), w.get("exam_weight",0)]
    vals.append(vals[0])
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats+[cats[0]], fill="toself",
        fillcolor="rgba(231,76,60,0.12)",
        line=dict(color="#e74c3c", width=2.5),
        marker=dict(size=9, color="#e74c3c")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,5], tickvals=[1,2,3,4,5]),
                   bgcolor="rgba(0,0,0,0)"),
        showlegend=False, margin=dict(t=20,b=20,l=50,r=50), height=260,
        font=dict(family="Noto Sans SC", size=12), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def render_bubble(kps):
    data = []
    for kp in kps:
        w = kp.get("weights", {})
        tier, _ = get_tier(w.get("core", 3))
        data.append({
            "知识点": kp["name"], "核心度": w.get("core",3), "难度": w.get("difficulty",3),
            "关联度": w.get("connectivity",3), "考试权重": w.get("exam_weight",3),
            "综合": w.get("core",3)+w.get("exam_weight",3), "等级": tier})
    fig = px.scatter(data, x="核心度", y="难度", size="综合", color="考试权重",
                     hover_name="知识点", size_max=35,
                     color_continuous_scale="YlOrRd",
                     labels={"考试权重":"📝 考试权重"})
    fig.update_layout(height=420, margin=dict(t=30,b=30),
                      font=dict(family="Noto Sans SC"),
                      xaxis=dict(range=[0.5,5.5], dtick=1, title="🎯 核心度"),
                      yaxis=dict(range=[0.5,5.5], dtick=1, title="🔥 难度"))
    return fig

# ================================================================
#               结构化 Markdown 兜底渲染
# ================================================================

def render_kps_as_markdown(kps):
    """当卡片渲染出问题时，用纯 Markdown 清晰展示"""
    lines = []
    cats = {}
    for kp in kps:
        cats.setdefault(kp.get("category", "未分类"), []).append(kp)

    for cat, items in cats.items():
        lines.append(f"\n### 📂 {cat}\n")
        for kp in items:
            w = kp.get("weights", {})
            tier, emoji = get_tier(w.get("core", 3))
            lines.append(f"**{emoji} {kp.get('id','')} {kp.get('name','')}** `{tier}级`\n")
            lines.append(f"> {kp.get('definition', '暂无定义')}\n")
            lines.append(
                f"🎯 核心`{w.get('core','?')}/5` · "
                f"🔥 难度`{w.get('difficulty','?')}/5` · "
                f"🔗 关联`{w.get('connectivity','?')}/5` · "
                f"📝 考试`{w.get('exam_weight','?')}/5` · "
                f"📄 `{kp.get('page','')}`\n")
            lines.append("---")
    return "\n".join(lines)


# ================================================================
#                     Prompt 模板
# ================================================================

PROMPT_TOC = """你是一个图书目录整理专家。以下是书籍前若干页的内容（已标注页码）。
请提取全局目录框架。

⚠️ 格式要求（严格遵守）：
- 每个章/节单独一行
- 章标题前加空行
- 每行必须标注页码范围
- 子节缩进 2 个空格

示例格式：

第1章 Introduction (p.1 - p.28)
  1.1 A First Problem: Stable Matching (p.1 - p.12)
  1.2 Five Representative Problems (p.12 - p.19)

第2章 Basics of Algorithm Analysis (p.29 - p.70)
  2.1 Computational Tractability (p.29 - p.35)

要求：
1. 每个条目必须标注页码范围
2. 不要省略任何章节
3. 只输出目录结构，不要输出正文
4. 章与章之间空一行

文本：
{toc_text}"""

PROMPT_EXERCISE = """基于以下知识点，针对【{query}】设计练习题：

🟢 基础题（1道）：围绕核心度最高的知识点，考查定义理解
🔴 深度思考题（1道）：围绕关联度最高的知识点，综合运用多个概念

格式：
### 🟢 基础题
**题目**：...
**提示**：...
**参考答案**：...

### 🔴 深度思考题
**题目**：...
**提示**：...
**参考答案**：...

知识点数据：
{content}"""


# ================================================================
#                     主流程
# ================================================================

uploaded_file = st.file_uploader("📎 上传书籍 PDF（支持大文件）", type="pdf")

if uploaded_file and api_key:
    if not st.session_state.processing_done:
        with st.spinner("📚 正在处理全书并生成目录..."):
            try:
                db, ptexts, toc_text, total = process_pdf(uploaded_file, toc_pages, chunk_size)
                st.session_state.vector_db = db
                st.session_state.page_texts = ptexts
                llm = get_llm()
                res = llm.invoke(PROMPT_TOC.format(toc_text=toc_text))
                st.session_state.framework = res.content
                st.session_state.processing_done = True
                st.rerun()
            except Exception as e:
                st.error(f"处理出错：{e}")
elif not api_key:
    st.info("👈 请先配置 API Key")

# ================================================================
#                  目录 + 提取界面
# ================================================================

if st.session_state.framework:
    col_toc, col_main = st.columns([1, 2])

    with col_toc:
        st.subheader("🌳 全局目录")
        st.info("💡 直接复制含页码的条目粘贴到右侧，精准提取零遗漏")
        # 格式化目录显示
        toc_md = st.session_state.framework
        st.markdown(toc_md)

    with col_main:
        st.subheader("🎯 深度提取 + 权重分析")
        chapter_query = st.text_input(
            "输入章节（建议含页码，如：`1.1 Stable Matching p.1-p.12`）：")

        opt1, opt2 = st.columns(2)
        with opt1:
            gen_exercises = st.checkbox("📝 生成练习题", value=False)
        with opt2:
            view_mode = st.selectbox("展示模式", ["🃏 卡片模式", "📋 Markdown 模式"])

        if st.button("🚀 开始提取与分析", use_container_width=True):
            if not chapter_query:
                st.warning("请先输入章节名称！")
            else:
                try:
                    llm = get_llm()
                    ps, pe = parse_pages(chapter_query)

                    if ps > 0:
                        st.info(f"📖 精确模式：第 {ps} - {pe} 页全文提取")
                        ctx = get_pages(ps, pe)
                        if not ctx.strip():
                            st.error("该页码范围无内容"); st.stop()
                        result = batch_extract(ctx, chapter_query, llm, max_chars_per_batch)
                    else:
                        st.warning("⚠️ 未检测到页码，语义搜索模式（建议复制含页码条目）")
                        ctx = sem_search(chapter_query, retrieve_k)
                        result = extract_single(ctx, chapter_query, llm)

                    if isinstance(result, list):
                        st.session_state.knowledge_points = result
                        st.session_state.raw_detail = json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        st.session_state.knowledge_points = None
                        st.session_state.raw_detail = result if isinstance(result, str) else str(result)

                    st.session_state.current_query = chapter_query

                    if gen_exercises:
                        with st.spinner("📝 生成练习题..."):
                            c = json.dumps(result, ensure_ascii=False) if isinstance(result, list) else str(result)
                            ex = llm.invoke(PROMPT_EXERCISE.format(query=chapter_query, content=c))
                            st.session_state.exercises = ex.content

                    st.rerun()
                except Exception as e:
                    st.error(f"出错：{e}")

# ================================================================
#                   结果可视化展示
# ================================================================

if st.session_state.knowledge_points:
    kps = st.session_state.knowledge_points
    query_label = st.session_state.current_query

    st.divider()
    st.subheader(f"📊 【{query_label}】知识点分析报告")

    # 统计面板
    render_stat_panel(kps)
    render_legend()

    # Tab 页
    tab_map, tab_cards, tab_bubble, tab_json = st.tabs([
        "🗺️ 知识图谱", "🃏 知识卡片", "📊 权重矩阵", "📄 原始数据"
    ])

    with tab_map:
        st.plotly_chart(render_treemap(kps), use_container_width=True)
        st.caption("方块面积 = 综合权重 · 颜色深浅 = 核心度 · 点击可下钻到子分类")

    with tab_cards:
        # 筛选栏
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            f_core = st.select_slider("⏬ 最低核心度", options=[1,2,3,4,5], value=1)
        with fc2:
            f_sort = st.selectbox("🔽 排序", ["核心度 ↓","难度 ↓","考试权重 ↓","页码 ↑"])
        with fc3:
            f_status = st.selectbox("🔘 状态", ["全部","未掌握","已掌握"])

        filtered = [k for k in kps if k.get("weights",{}).get("core",0) >= f_core]
        if f_status == "未掌握":
            filtered = [k for k in filtered if not st.session_state.mastered.get(k.get("id",""), False)]
        elif f_status == "已掌握":
            filtered = [k for k in filtered if st.session_state.mastered.get(k.get("id",""), False)]

        sort_map = {
            "核心度 ↓": lambda k: -k.get("weights",{}).get("core",0),
            "难度 ↓": lambda k: -k.get("weights",{}).get("difficulty",0),
            "考试权重 ↓": lambda k: -k.get("weights",{}).get("exam_weight",0),
            "页码 ↑": lambda k: int(re.search(r"\d+", k.get("page","0")).group()) if re.search(r"\d+", k.get("page","0")) else 0,
        }
        filtered.sort(key=sort_map.get(f_sort, sort_map["核心度 ↓"]))
        st.caption(f"📋 显示 {len(filtered)}/{len(kps)} 个知识点")

        for i, kp in enumerate(filtered):
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(render_card_html(kp), unsafe_allow_html=True)
            with c2:
                kid = kp.get("id", f"kp-{i}")
                mastered = st.checkbox("✅", value=st.session_state.mastered.get(kid, False),
                                       key=f"m_{kid}", help="标记已掌握")
                st.session_state.mastered[kid] = mastered

            with st.expander(f"📊 {kp['name']} 详细分析"):
                rc1, rc2 = st.columns([1, 1])
                with rc1:
                    st.plotly_chart(render_radar(kp), use_container_width=True)
                with rc2:
                    w = kp.get("weights", {})
                    total = sum(w.values())
                    tier, emoji = get_tier(w.get("core",0))
                    st.markdown(f"""
#### {emoji} 综合评分 {total}/20 · {tier}级

| 维度 | 分数 | 解读 |
|------|------|------|
| 🎯 核心度 | {w.get('core',0)}/5 | {"**必须掌握**" if w.get('core',0)>=4 else "建议了解" if w.get('core',0)>=3 else "可选"} |
| 🔥 难度 | {w.get('difficulty',0)}/5 | {"**重点攻克**" if w.get('difficulty',0)>=4 else "中等" if w.get('difficulty',0)>=3 else "容易"} |
| 🔗 关联度 | {w.get('connectivity',0)}/5 | {"**枢纽节点**" if w.get('connectivity',0)>=4 else "有关联" if w.get('connectivity',0)>=3 else "独立"} |
| 📝 考试 | {w.get('exam_weight',0)}/5 | {"**高频考点**" if w.get('exam_weight',0)>=4 else "可能出题" if w.get('exam_weight',0)>=3 else "低频"} |
""")
                    pre = kp.get("prerequisites", [])
                    rel = kp.get("related", [])
                    if pre: st.markdown(f"**🔙 前置知识**: {', '.join(pre)}")
                    if rel: st.markdown(f"**🔀 相关知识**: {', '.join(rel)}")

    with tab_bubble:
        st.plotly_chart(render_bubble(kps), use_container_width=True)
        st.caption("X轴 = 核心度 · Y轴 = 难度 · 气泡大小 = 综合分 · 颜色 = 考试权重（越红越高）")
        st.markdown("💡 **右上角红色大气泡** = 又核心、又难、又常考 → 优先攻克目标")

    with tab_json:
        st.code(st.session_state.raw_detail, language="json")

    # 练习题
    if st.session_state.exercises:
        st.divider()
        st.subheader("📝 练习题")
        st.markdown(st.session_state.exercises)

elif st.session_state.raw_detail and not st.session_state.knowledge_points:
    # ===== 兜底：JSON 解析彻底失败，用格式化 Markdown 展示 =====
    st.divider()
    st.subheader(f"📄 【{st.session_state.current_query}】提取结果")
    st.warning("⚠️ 结构化解析未成功，以文本模式展示。建议重新提取或调整参数。")

    raw = st.session_state.raw_detail
    # 最后尝试一次解析
    last_try = robust_parse_json(raw)
    if last_try:
        st.session_state.knowledge_points = last_try
        st.rerun()
    else:
        st.markdown(raw)

    if st.session_state.exercises:
        st.divider()
        st.subheader("📝 练习题")
        st.markdown(st.session_state.exercises)
