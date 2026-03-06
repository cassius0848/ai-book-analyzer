import re
import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# ========== 页面设置 ==========
st.set_page_config(page_title="AI 文件整理解析器", layout="wide")
st.title("📖 知识点全量提取助手 (大文件架构)")
st.caption("采用"分而治之"架构：左侧生成全局目录，右侧按需提取详细定义与习题")

# ========== 初始化 Session State ==========
defaults = {
    "vector_db": None,
    "page_texts": {},       # {页码: 原文}，全书逐页原文存储
    "framework": "",
    "llm": None,
    "processing_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ========== 侧边栏：配置中心 ==========
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
    toc_pages = st.slider("目录扫描页数", min_value=5, max_value=30, value=15,
                          help="大多数书的目录集中在前 10-20 页")
    chunk_size = st.slider("文本块大小", min_value=500, max_value=2000, value=1000, step=100)
    retrieve_k = st.slider("语义检索块数 (k)", min_value=4, max_value=30, value=15,
                           help="仅在无页码时使用，越大覆盖越全")
    max_chars_per_batch = st.slider(
        "单次提取最大字符数", min_value=4000, max_value=30000, value=12000, step=2000,
        help="章节原文超过此长度时，自动分批提取再合并"
    )


# ========== 核心函数 ==========

@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    if st.session_state.llm is None:
        st.session_state.llm = ChatOpenAI(
            model=model_name, api_key=api_key, base_url=base_url
        )
    return st.session_state.llm


def process_pdf(file, toc_pages_count: int, chunk_sz: int):
    """
    处理 PDF：
    1. 逐页存储原文到 page_texts（用于精确提取，零遗漏）
    2. 逐页切块存入 FAISS（用于无页码时的语义搜索兜底）
    3. 前 N 页带页码锚点，用于目录提取
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    toc_text = ""
    page_texts = {}
    all_chunks = []
    all_metadatas = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_sz, chunk_overlap=150)
    embeddings = load_embedding_model()

    for i, page in enumerate(doc):
        content = page.get_text()
        page_num = i + 1
        page_texts[page_num] = content

        if i < toc_pages_count:
            toc_text += f"\n--- 第{page_num}页 ---\n{content}"

        chunks = splitter.split_text(content)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"page": page_num})

    vector_db = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)
    return vector_db, page_texts, toc_text, len(doc)


# ---------- 检索策略 ----------

def get_full_text_by_pages(page_start: int, page_end: int) -> str:
    """精确提取：直接拼接指定页码范围的全部原文，零遗漏。"""
    page_texts = st.session_state.page_texts
    parts = []
    for p in range(page_start, page_end + 1):
        if p in page_texts:
            parts.append(f"[第{p}页]\n{page_texts[p]}")
    return "\n\n".join(parts)


def semantic_fallback(query: str, k: int) -> str:
    """兜底方案：无页码时用语义搜索。"""
    docs = st.session_state.vector_db.similarity_search(query, k=k)
    docs.sort(key=lambda d: d.metadata.get("page", 0))
    return "\n".join(
        [f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
    )


def parse_page_range(user_input: str):
    """
    从用户输入中解析页码范围，支持多种格式：
    p.10-p.30 / p10-30 / 第10页-第30页 / p.10 等
    """
    matches = re.findall(r"p\.?\s*(\d+)", user_input)
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    if len(matches) == 1:
        return int(matches[0]), int(matches[0]) + 20

    matches = re.findall(r"第\s*(\d+)\s*页", user_input)
    if len(matches) >= 2:
        return int(matches[0]), int(matches[1])
    if len(matches) == 1:
        return int(matches[0]), int(matches[0]) + 20

    return 0, 0


# ---------- 分批 Map-Reduce ----------

def batch_extract(full_text: str, query: str, llm, max_chars: int) -> str:
    """
    章节原文过长时，分批提取再合并，确保零遗漏。
    短文本直接单次处理。
    """
    if len(full_text) <= max_chars:
        prompt = PROMPT_DETAIL.format(query=query, context=full_text)
        return llm.invoke(prompt).content

    # --- 按页块分批，不在页中间切断 ---
    pages = full_text.split("\n\n")
    batches = []
    current_batch = ""

    for page_block in pages:
        if len(current_batch) + len(page_block) > max_chars and current_batch:
            batches.append(current_batch)
            current_batch = page_block
        else:
            current_batch += ("\n\n" + page_block) if current_batch else page_block
    if current_batch:
        batches.append(current_batch)

    # --- Map: 逐批提取 ---
    batch_results = []
    progress = st.progress(0, text="分批提取中...")
    for i, batch in enumerate(batches):
        progress.progress((i + 1) / (len(batches) + 1), text=f"正在处理第 {i+1}/{len(batches)} 批...")
        prompt = f"""你是一位严谨的教研专家。用户想学习【{query}】。
以下是该章节的第 {i+1}/{len(batches)} 部分原文（方括号内为页码）。

【任务】：
1. 逐页扫描，提取这部分中出现的所有重要概念、定理、公式、专业名词
2. 为每个专业名词给出精准定义（忠于原文）
3. 在每个知识点后标注来源页码 [p.XX]
4. 不要添加原文没有的内容

【原文片段】：
{batch}"""
        result = llm.invoke(prompt).content
        batch_results.append(result)

    # --- Reduce: 合并去重 ---
    progress.progress(1.0, text="正在合并所有批次...")
    all_batch_text = "\n\n---\n\n".join(
        [f"【第{i+1}批提取结果】\n{r}" for i, r in enumerate(batch_results)]
    )

    reduce_prompt = f"""你是一位严谨的教研专家。我已分 {len(batches)} 批提取了【{query}】的知识点。
请合并为一份完整的学习框架。

【合并要求】：
1. 去除重复内容，保留最完整的版本
2. 按原书逻辑顺序重新排列（参考页码）
3. 确保所有专业名词、定义、公式都保留，不遗漏
4. 保留每个知识点的页码标注
5. 输出结构清晰、层次分明的最终版本

【各批次结果】：
{all_batch_text}"""

    final = llm.invoke(reduce_prompt).content
    progress.empty()
    return final


# ========== Prompt 模板 ==========

PROMPT_TOC = """你是一个图书目录整理专家。以下是书籍前若干页的内容（已标注页码）。
请提取全局目录框架，格式严格如下：

第1章 章节名称 (p.X - p.Y)
  1.1 小节名称 (p.X - p.Y)
  1.2 小节名称 (p.X - p.Y)
第2章 章节名称 (p.X - p.Y)
  ...

要求：
1. 每个条目必须标注页码范围（根据目录页的页码信息推断）
2. 不要省略任何章节
3. 只输出目录结构，不要输出正文内容

文本：
{toc_text}"""

PROMPT_DETAIL = """你是一位严谨的教研专家。用户是初学者，想深度学习【{query}】。
以下是该章节的完整原文（方括号内为页码）。

【核心原则】：你拿到的是该章节的全部内容，请逐页扫描，不要跳过任何知识点。

【任务要求】：
1. 逐页扫描原文，提取所有出现的重要概念、定理、公式、专业名词
2. 为每个专业名词给出精准定义（忠于原文表述）
3. 按原书的逻辑顺序组织，形成层次分明的知识框架
4. 如果某页包含图表说明或例题，也要提取其核心结论
5. 在每个知识点后标注来源页码，如 [p.23]
6. 绝对忠于原文，不要添加原文没有的内容；如果原文未涉及某概念，明确说明

【该章节完整原文】：
{context}"""

PROMPT_EXERCISE = """基于以下学习内容，请针对【{query}】设计练习题：

1. 🟢 基础题（1道）：考查核心定义和基本概念的理解
2. 🔴 深度思考题（1道）：需要综合运用多个概念，或联系实际场景

每道题给出：题目 → 提示 → 参考答案

学习内容：
{content}"""


# ========== 主界面：上传与处理 ==========

uploaded_file = st.file_uploader("上传书籍 PDF（支持大文件）", type="pdf")

if uploaded_file and api_key:
    if not st.session_state.processing_done:
        with st.spinner("📚 正在切片整本书并提取全局目录，请稍候..."):
            try:
                db, page_texts, toc_text, total_pages = process_pdf(
                    uploaded_file, toc_pages, chunk_size
                )
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
    st.info("👈 请先在左侧配置 API Key")


# ========== 结果展示与深度提取 ==========

if st.session_state.framework:
    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("🌳 全局目录树")
        st.info("✨ 直接复制目录中的条目（含页码）粘贴到右侧，即可精准提取全部内容，零遗漏。")
        st.markdown(st.session_state.framework)

    with col2:
        st.subheader("🎯 按需深度提取")

        chapter_query = st.text_input(
            "输入想学习的部分（建议直接从左侧复制含页码的条目）："
        )

        generate_exercises = st.checkbox("📝 同时生成练习题", value=False)

        if st.button("🚀 开始提取"):
            if not chapter_query:
                st.warning("请先输入章节名称！")
            else:
                with st.spinner(f"正在提取【{chapter_query}】..."):
                    try:
                        llm = get_llm()

                        # 1. 解析页码
                        page_start, page_end = parse_page_range(chapter_query)

                        # 2. 选择检索策略
                        if page_start > 0:
                            # ✅ 有页码 → 直接取全部原文，零遗漏
                            st.info(f"📖 精确模式：提取第 {page_start} - {page_end} 页全部原文")
                            context = get_full_text_by_pages(page_start, page_end)

                            if not context.strip():
                                st.error(f"第 {page_start}-{page_end} 页没有内容，请检查页码范围。")
                                st.stop()

                            # 根据长度选择单次 or 分批 Map-Reduce
                            detail_content = batch_extract(
                                context, chapter_query, llm, max_chars_per_batch
                            )
                        else:
                            # ⚠️ 无页码 → 语义搜索兜底
                            st.warning("⚠️ 未检测到页码，使用语义搜索（可能有遗漏）。建议从左侧复制含页码的条目。")
                            context = semantic_fallback(chapter_query, retrieve_k)
                            detail_content = llm.invoke(
                                PROMPT_DETAIL.format(query=chapter_query, context=context)
                            ).content

                        st.success(f"✅ 【{chapter_query}】提取完成！")
                        st.markdown(detail_content)

                        # 可选：生成练习题
                        if generate_exercises:
                            st.divider()
                            with st.spinner("📝 正在生成练习题..."):
                                exercise_result = llm.invoke(
                                    PROMPT_EXERCISE.format(
                                        query=chapter_query,
                                        content=detail_content,
                                    )
                                )
                                st.subheader("📝 练习题")
                                st.markdown(exercise_result.content)

                    except Exception as e:
                        st.error(f"提取出错：{e}")
