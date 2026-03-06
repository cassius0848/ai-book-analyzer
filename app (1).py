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

# ========== 页面设置 ==========
st.set_page_config(page_title="AI 知识提取器", layout="wide")

# ========== 自定义样式 ==========
st.markdown("""
<style>
/* 全局字体 */
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }

/* 知识卡片 */
.kp-card {
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    border-left: 5px solid;
    background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kp-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.1);
}
.kp-card.core-5 { border-left-color: #e74c3c; background: linear-gradient(135deg, #fef2f2 0%, #fff 100%); }
.kp-card.core-4 { border-left-color: #f39c12; background: linear-gradient(135deg, #fffbeb 0%, #fff 100%); }
.kp-card.core-3 { border-left-color: #3498db; background: linear-gradient(135deg, #eff6ff 0%, #fff 100%); }
.kp-card.core-2 { border-left-color: #2ecc71; background: linear-gradient(135deg, #f0fdf4 0%, #fff 100%); }
.kp-card.core-1 { border-left-color: #95a5a6; background: linear-gradient(135deg, #f9fafb 0%, #fff 100%); }

.kp-title {
    font-size: 1.05em;
    font-weight: 700;
    margin-bottom: 6px;
    color: #1a1a2e;
}
.kp-def {
    font-size: 0.92em;
    color: #374151;
    line-height: 1.6;
    margin-bottom: 10px;
}
.kp-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.kp-badge {
    font-size: 0.75em;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 500;
    color: #fff;
}
.badge-core { background: #e74c3c; }
.badge-diff { background: #8e44ad; }
.badge-conn { background: #2980b9; }
.badge-exam { background: #27ae60; }
.badge-page { background: #7f8c8d; }

/* 权重图例 */
.legend-row {
    display: flex; align-items: center; gap: 20px;
    padding: 10px 0; flex-wrap: wrap;
}
.legend-item {
    display: flex; align-items: center; gap: 6px; font-size: 0.85em; color: #555;
}
.legend-dot {
    width: 12px; height: 12px; border-radius: 3px;
}

/* 统计面板 */
.stat-box {
    text-align: center;
    padding: 16px;
    border-radius: 12px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.stat-num { font-size: 2em; font-weight: 700; }
.stat-label { font-size: 0.85em; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

st.title("📖 知识点全量提取助手")
st.caption("分而治之 · 权重分析 · 可视化知识图谱")

# ========== Session State ==========
defaults = {
    "vector_db": None,
    "page_texts": {},
    "framework": "",
    "llm": None,
    "processing_done": False,
    "knowledge_points": None,   # 结构化知识点数据
    "raw_detail": "",           # 原始文本输出（兜底）
    "mastered": {},             # 用户标记的已掌握知识点
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ========== 侧边栏 ==========
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


# ========== 核心函数 ==========

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_llm():
    if st.session_state.llm is None:
        st.session_state.llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
    return st.session_state.llm

def process_pdf(file, toc_pages_count, chunk_sz):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    toc_text, page_texts = "", {}
    all_chunks, all_metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_sz, chunk_overlap=150)
    embeddings = load_embedding_model()

    for i, page in enumerate(doc):
        content = page.get_text()
        page_num = i + 1
        page_texts[page_num] = content
        if i < toc_pages_count:
            toc_text += f"\n--- 第{page_num}页 ---\n{content}"
        for chunk in splitter.split_text(content):
            all_chunks.append(chunk)
            all_metadatas.append({"page": page_num})

    vector_db = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)
    return vector_db, page_texts, toc_text, len(doc)


# ---------- 检索 ----------

def get_full_text_by_pages(ps, pe):
    pts = st.session_state.page_texts
    return "\n\n".join([f"[第{p}页]\n{pts[p]}" for p in range(ps, pe + 1) if p in pts])

def semantic_fallback(query, k):
    docs = st.session_state.vector_db.similarity_search(query, k=k)
    docs.sort(key=lambda d: d.metadata.get("page", 0))
    return "\n".join([f"[p.{d.metadata.get('page','?')}] {d.page_content}" for d in docs])

def parse_page_range(text):
    m = re.findall(r"p\.?\s*(\d+)", text)
    if len(m) >= 2: return int(m[0]), int(m[1])
    if len(m) == 1: return int(m[0]), int(m[0]) + 20
    m = re.findall(r"第\s*(\d+)\s*页", text)
    if len(m) >= 2: return int(m[0]), int(m[1])
    if len(m) == 1: return int(m[0]), int(m[0]) + 20
    return 0, 0


# ---------- 分批 Map-Reduce ----------

def batch_extract(full_text, query, llm, max_chars):
    if len(full_text) <= max_chars:
        return extract_single(full_text, query, llm)

    pages = full_text.split("\n\n")
    batches, cur = [], ""
    for block in pages:
        if len(cur) + len(block) > max_chars and cur:
            batches.append(cur); cur = block
        else:
            cur += ("\n\n" + block) if cur else block
    if cur: batches.append(cur)

    batch_results = []
    progress = st.progress(0, text="分批提取中...")
    for i, batch in enumerate(batches):
        progress.progress((i + 1) / (len(batches) + 1), text=f"第 {i+1}/{len(batches)} 批...")
        result = extract_single(batch, query, llm, batch_label=f"{i+1}/{len(batches)}")
        batch_results.append(result)

    progress.progress(1.0, text="合并中...")

    # Reduce: 合并所有批次的 JSON
    all_kps = []
    for r in batch_results:
        if isinstance(r, list):
            all_kps.extend(r)
        elif isinstance(r, str):
            return r  # fallback 纯文本

    # 去重 + 让 AI 做最终整理
    reduce_prompt = f"""你是教研专家。我分批提取了【{query}】的知识点（JSON 格式）。
请合并去重，输出最终版本。

【要求】：
1. 合并重复知识点，保留信息最完整的版本
2. 重新审视权重评分，确保整体一致性
3. 严格输出 JSON 数组，不要输出其他任何文字

JSON 格式（每个元素）：
{{
  "id": "唯一编号如 KP-01",
  "name": "知识点名称",
  "category": "所属小节/主题",
  "definition": "精准定义（忠于原文）",
  "page": "来源页码如 p.23",
  "weights": {{
    "core": 1-5,
    "difficulty": 1-5,
    "connectivity": 1-5,
    "exam_weight": 1-5
  }},
  "prerequisites": ["前置知识点ID"],
  "related": ["相关知识点ID"]
}}

待合并数据：
{json.dumps(all_kps, ensure_ascii=False, indent=2)}"""

    resp = llm.invoke(reduce_prompt).content
    progress.empty()
    return parse_kp_json(resp)


def extract_single(text, query, llm, batch_label=None):
    """单次提取，要求 AI 输出结构化 JSON + 权重评分"""
    batch_note = f"（这是第{batch_label}部分）" if batch_label else ""

    prompt = f"""你是一位严谨的教研专家。用户是初学者，想深度学习【{query}】。
以下是该章节的原文{batch_note}（方括号内为页码）。

【核心任务】：逐页扫描原文，提取所有重要知识点，并为每个知识点进行多维权重评估。

【权重评估维度】（每项 1-5 分）：
- core（核心度）：该知识点在本章中的基础重要性。5=绝对核心概念，1=补充性细节
- difficulty（难度）：初学者理解的困难程度。5=非常抽象/复杂，1=直观易懂
- connectivity（关联度）：与其他知识点的关联紧密程度。5=高度关联多个概念，1=较独立
- exam_weight（考试权重）：出现在考试/练习中的可能性。5=必考重点，1=了解即可

【输出格式】：严格输出 JSON 数组，不要输出任何其他文字、解释、markdown标记。
每个元素格式如下：
{{
  "id": "KP-01",
  "name": "知识点名称（中英文皆可）",
  "category": "所属小节或主题分类",
  "definition": "忠于原文的精准定义（2-4句话）",
  "page": "p.23",
  "weights": {{
    "core": 4,
    "difficulty": 3,
    "connectivity": 5,
    "exam_weight": 4
  }},
  "prerequisites": ["前置知识点的ID，如KP-02"],
  "related": ["相关知识点的ID"]
}}

【原文】：
{text}"""

    resp = llm.invoke(prompt).content
    return parse_kp_json(resp)


def parse_kp_json(text):
    """从 AI 回复中提取 JSON，容错处理"""
    # 去掉 markdown 代码块标记
    cleaned = re.sub(r"```json\s*", "", text)
    cleaned = re.sub(r"```\s*", "", cleaned)
    cleaned = cleaned.strip()

    # 尝试找到 JSON 数组
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list) and len(data) > 0:
                return data
        except json.JSONDecodeError:
            pass

    # 解析失败，返回原文
    return text


# ========== 可视化函数 ==========

def render_treemap(kps):
    """知识点层级 Treemap：大小=综合权重，颜色=核心度"""
    names, parents, values, colors, hovers = [], [], [], [], []

    # 按 category 分组
    categories = {}
    for kp in kps:
        cat = kp.get("category", "未分类")
        categories.setdefault(cat, []).append(kp)

    root = "知识图谱"
    names.append(root); parents.append(""); values.append(0); colors.append(0); hovers.append("")

    for cat, items in categories.items():
        names.append(cat); parents.append(root); values.append(0); colors.append(0); hovers.append(f"主题: {cat}")
        for kp in items:
            w = kp.get("weights", {})
            total = w.get("core", 3) + w.get("difficulty", 3) + w.get("connectivity", 3) + w.get("exam_weight", 3)
            names.append(kp["name"])
            parents.append(cat)
            values.append(total)
            colors.append(w.get("core", 3))
            hovers.append(
                f"<b>{kp['name']}</b><br>"
                f"核心度: {'⭐' * w.get('core', 0)}<br>"
                f"难度: {'🔥' * w.get('difficulty', 0)}<br>"
                f"关联度: {'🔗' * w.get('connectivity', 0)}<br>"
                f"考试权重: {'📝' * w.get('exam_weight', 0)}<br>"
                f"来源: {kp.get('page', '?')}"
            )

    fig = go.Figure(go.Treemap(
        labels=names,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=[[0, "#a8d8ea"], [0.5, "#fcbad3"], [1, "#e74c3c"]],
            line=dict(width=2, color="white"),
        ),
        hovertext=hovers,
        hoverinfo="text",
        textinfo="label",
        textfont=dict(size=14),
        pathbar=dict(visible=True),
    ))
    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        height=450,
        font=dict(family="Noto Sans SC"),
    )
    return fig


def render_radar(kp):
    """单个知识点的权重雷达图"""
    w = kp.get("weights", {})
    categories_r = ["核心度", "难度", "关联度", "考试权重"]
    values_r = [w.get("core", 0), w.get("difficulty", 0), w.get("connectivity", 0), w.get("exam_weight", 0)]
    values_r.append(values_r[0])  # 闭合

    fig = go.Figure(go.Scatterpolar(
        r=values_r,
        theta=categories_r + [categories_r[0]],
        fill="toself",
        fillcolor="rgba(231, 76, 60, 0.15)",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=8, color="#e74c3c"),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], tickvals=[1, 2, 3, 4, 5]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=40, r=40),
        height=250,
        font=dict(family="Noto Sans SC", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_weight_distribution(kps):
    """全部知识点的四维权重分布气泡图"""
    data = []
    for kp in kps:
        w = kp.get("weights", {})
        data.append({
            "知识点": kp["name"],
            "核心度": w.get("core", 3),
            "难度": w.get("difficulty", 3),
            "关联度": w.get("connectivity", 3),
            "考试权重": w.get("exam_weight", 3),
            "综合分": w.get("core", 3) + w.get("exam_weight", 3),
        })

    fig = px.scatter(
        data, x="核心度", y="难度",
        size="综合分", color="考试权重",
        hover_name="知识点",
        size_max=30,
        color_continuous_scale="RdYlGn_r",
        labels={"考试权重": "考试权重"},
    )
    fig.update_layout(
        height=400,
        margin=dict(t=30, b=30),
        font=dict(family="Noto Sans SC"),
        xaxis=dict(range=[0.5, 5.5], dtick=1),
        yaxis=dict(range=[0.5, 5.5], dtick=1),
    )
    return fig


def render_knowledge_card(kp, index):
    """渲染单个知识卡片 HTML"""
    w = kp.get("weights", {})
    core = w.get("core", 3)
    card_class = f"core-{min(core, 5)}"

    html = f"""
    <div class="kp-card {card_class}">
        <div class="kp-title">{kp.get('id', '')} · {kp['name']}</div>
        <div class="kp-def">{kp.get('definition', '暂无定义')}</div>
        <div class="kp-badges">
            <span class="kp-badge badge-core">核心 {'⭐' * core}</span>
            <span class="kp-badge badge-diff">难度 {w.get('difficulty', '?')}/5</span>
            <span class="kp-badge badge-conn">关联 {w.get('connectivity', '?')}/5</span>
            <span class="kp-badge badge-exam">考试 {w.get('exam_weight', '?')}/5</span>
            <span class="kp-badge badge-page">{kp.get('page', '')}</span>
        </div>
    </div>
    """
    return html


# ========== Prompt 模板 ==========

PROMPT_TOC = """你是一个图书目录整理专家。以下是书籍前若干页的内容（已标注页码）。
请提取全局目录框架，格式严格如下：

第1章 章节名称 (p.X - p.Y)
  1.1 小节名称 (p.X - p.Y)
  1.2 小节名称 (p.X - p.Y)
第2章 章节名称 (p.X - p.Y)
  ...

要求：
1. 每个条目必须标注页码范围
2. 不要省略任何章节
3. 只输出目录结构

文本：
{toc_text}"""

PROMPT_EXERCISE = """基于以下知识点数据，请针对【{query}】设计练习题：

1. 🟢 基础题（1道）：围绕核心度最高的知识点，考查定义理解
2. 🔴 深度思考题（1道）：围绕关联度最高的知识点，需综合运用多个概念

每道题给出：题目 → 提示 → 参考答案

知识点数据：
{content}"""


# ========== 主界面 ==========

uploaded_file = st.file_uploader("上传书籍 PDF（支持大文件）", type="pdf")

if uploaded_file and api_key:
    if not st.session_state.processing_done:
        with st.spinner("📚 正在处理全书并生成目录..."):
            try:
                db, page_texts, toc_text, total = process_pdf(uploaded_file, toc_pages, chunk_size)
                st.session_state.vector_db = db
                st.session_state.page_texts = page_texts
                llm = get_llm()
                result = llm.invoke(PROMPT_TOC.format(toc_text=toc_text))
                st.session_state.framework = result.content
                st.session_state.processing_done = True
                st.rerun()
            except Exception as e:
                st.error(f"处理出错：{e}")
elif not api_key:
    st.info("👈 请先配置 API Key")


# ========== 目录 + 提取界面 ==========

if st.session_state.framework:
    col_toc, col_main = st.columns([1, 2])

    with col_toc:
        st.subheader("🌳 全局目录")
        st.info("✨ 复制含页码的条目到右侧，精准提取零遗漏")
        st.markdown(st.session_state.framework)

    with col_main:
        st.subheader("🎯 深度提取 + 权重分析")

        chapter_query = st.text_input("输入章节（建议含页码，如：1.2 向量空间 p.30-p.55）：")

        opt_col1, opt_col2 = st.columns(2)
        with opt_col1:
            generate_exercises = st.checkbox("📝 生成练习题", value=False)
        with opt_col2:
            show_raw_text = st.checkbox("📄 同时显示原始文本输出", value=False)

        if st.button("🚀 开始提取与分析", use_container_width=True):
            if not chapter_query:
                st.warning("请先输入章节名称！")
            else:
                try:
                    llm = get_llm()
                    ps, pe = parse_page_range(chapter_query)

                    if ps > 0:
                        st.info(f"📖 精确模式：第 {ps} - {pe} 页")
                        context = get_full_text_by_pages(ps, pe)
                        if not context.strip():
                            st.error("该页码范围无内容，请检查。"); st.stop()
                        result = batch_extract(context, chapter_query, llm, max_chars_per_batch)
                    else:
                        st.warning("⚠️ 未检测到页码，语义搜索模式（可能有遗漏）")
                        context = semantic_fallback(chapter_query, retrieve_k)
                        result = extract_single(context, chapter_query, llm)

                    # 存储结果
                    if isinstance(result, list):
                        st.session_state.knowledge_points = result
                        st.session_state.raw_detail = json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        st.session_state.knowledge_points = None
                        st.session_state.raw_detail = result

                    st.success(f"✅ 【{chapter_query}】提取完成！")

                    # 练习题
                    if generate_exercises:
                        with st.spinner("📝 生成练习题..."):
                            content_for_ex = json.dumps(result, ensure_ascii=False) if isinstance(result, list) else result
                            ex = llm.invoke(PROMPT_EXERCISE.format(query=chapter_query, content=content_for_ex))
                            st.session_state["exercises"] = ex.content

                    st.rerun()

                except Exception as e:
                    st.error(f"出错：{e}")


# ========== 可视化结果展示 ==========

if st.session_state.knowledge_points:
    kps = st.session_state.knowledge_points

    st.divider()

    # --- 统计面板 ---
    s1, s2, s3, s4 = st.columns(4)
    core_5 = len([k for k in kps if k.get("weights", {}).get("core", 0) >= 4])
    avg_diff = sum(k.get("weights", {}).get("difficulty", 0) for k in kps) / max(len(kps), 1)
    mastered_count = sum(1 for k in kps if st.session_state.mastered.get(k.get("id", ""), False))

    with s1:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{len(kps)}</div><div class="stat-label">知识点总数</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{core_5}</div><div class="stat-label">核心知识点</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{avg_diff:.1f}</div><div class="stat-label">平均难度</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-box"><div class="stat-num">{mastered_count}/{len(kps)}</div><div class="stat-label">已掌握</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- 图例 ---
    st.markdown("""
    <div class="legend-row">
        <div class="legend-item"><div class="legend-dot" style="background:#e74c3c"></div>核心度 5 (必须掌握)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#f39c12"></div>核心度 4 (重要)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#3498db"></div>核心度 3 (一般)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#2ecc71"></div>核心度 2 (了解)</div>
        <div class="legend-item"><div class="legend-dot" style="background:#95a5a6"></div>核心度 1 (补充)</div>
    </div>
    """, unsafe_allow_html=True)

    # --- Tab 页 ---
    tab_map, tab_cards, tab_bubble, tab_raw = st.tabs(["🗺️ 知识图谱", "🃏 知识卡片", "📊 权重分布", "📄 原始输出"])

    with tab_map:
        st.plotly_chart(render_treemap(kps), use_container_width=True)

    with tab_cards:
        # 筛选控件
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_core = st.select_slider("最低核心度", options=[1, 2, 3, 4, 5], value=1)
        with fc2:
            sort_by = st.selectbox("排序", ["核心度 ↓", "难度 ↓", "考试权重 ↓", "页码"])
        with fc3:
            filter_mastered = st.selectbox("掌握状态", ["全部", "未掌握", "已掌握"])

        # 过滤
        filtered = [k for k in kps if k.get("weights", {}).get("core", 0) >= filter_core]
        if filter_mastered == "未掌握":
            filtered = [k for k in filtered if not st.session_state.mastered.get(k.get("id", ""), False)]
        elif filter_mastered == "已掌握":
            filtered = [k for k in filtered if st.session_state.mastered.get(k.get("id", ""), False)]

        # 排序
        sort_keys = {
            "核心度 ↓": lambda k: -k.get("weights", {}).get("core", 0),
            "难度 ↓": lambda k: -k.get("weights", {}).get("difficulty", 0),
            "考试权重 ↓": lambda k: -k.get("weights", {}).get("exam_weight", 0),
            "页码": lambda k: int(re.search(r"\d+", k.get("page", "0")).group()) if re.search(r"\d+", k.get("page", "0")) else 0,
        }
        filtered.sort(key=sort_keys.get(sort_by, sort_keys["核心度 ↓"]))

        st.caption(f"显示 {len(filtered)}/{len(kps)} 个知识点")

        for i, kp in enumerate(filtered):
            card_col, action_col = st.columns([5, 1])
            with card_col:
                st.markdown(render_knowledge_card(kp, i), unsafe_allow_html=True)
            with action_col:
                kp_id = kp.get("id", f"kp-{i}")
                is_mastered = st.session_state.mastered.get(kp_id, False)
                if st.checkbox("✅", value=is_mastered, key=f"master_{kp_id}",
                               help="标记为已掌握"):
                    st.session_state.mastered[kp_id] = True
                else:
                    st.session_state.mastered[kp_id] = False

            # 展开查看雷达图
            with st.expander(f"📊 查看 {kp['name']} 的权重雷达图"):
                r_col1, r_col2 = st.columns([1, 1])
                with r_col1:
                    st.plotly_chart(render_radar(kp), use_container_width=True)
                with r_col2:
                    w = kp.get("weights", {})
                    total = sum(w.values())
                    st.markdown(f"""
**综合评分**：{total}/20

**权重解读**：
- 🎯 核心度 {w.get('core',0)}/5 — {"必须掌握" if w.get('core',0)>=4 else "建议了解" if w.get('core',0)>=3 else "可选"}
- 🔥 难度 {w.get('difficulty',0)}/5 — {"需要重点攻克" if w.get('difficulty',0)>=4 else "中等" if w.get('difficulty',0)>=3 else "容易上手"}
- 🔗 关联度 {w.get('connectivity',0)}/5 — {"枢纽知识点" if w.get('connectivity',0)>=4 else "有关联" if w.get('connectivity',0)>=3 else "较独立"}
- 📝 考试 {w.get('exam_weight',0)}/5 — {"高频考点" if w.get('exam_weight',0)>=4 else "可能出题" if w.get('exam_weight',0)>=3 else "低频"}
""")
                    prereqs = kp.get("prerequisites", [])
                    related = kp.get("related", [])
                    if prereqs:
                        st.markdown(f"**前置知识**: {', '.join(prereqs)}")
                    if related:
                        st.markdown(f"**相关知识**: {', '.join(related)}")

    with tab_bubble:
        st.plotly_chart(render_weight_distribution(kps), use_container_width=True)
        st.caption("X轴=核心度，Y轴=难度，气泡大小=核心度+考试权重，颜色=考试权重（红=高）")

    with tab_raw:
        if show_raw_text or True:
            st.code(st.session_state.raw_detail, language="json")

    # --- 练习题展示 ---
    if st.session_state.get("exercises"):
        st.divider()
        st.subheader("📝 练习题")
        st.markdown(st.session_state["exercises"])

elif st.session_state.raw_detail and not st.session_state.knowledge_points:
    # JSON 解析失败的兜底：直接显示文本
    st.divider()
    st.subheader("📄 提取结果（文本模式）")
    st.markdown(st.session_state.raw_detail)
