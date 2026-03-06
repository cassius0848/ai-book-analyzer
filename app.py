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
        st.success("✅ 已自动加载内部测试 API Key")
    except Exception:
        api_key = st.text_input("输入 DeepSeek API Key", type="password")

    base_url = st.text_input("API 代理地址", value="https://api.deepseek.com")
    model_name = st.selectbox("选择模型", ["deepseek-chat", "deepseek-reasoner"])

    st.divider()
    st.header("📐 高级参数")
    toc_pages = st.slider("目录扫描页数", min_value=5, max_value=30, value=15,
                          help="大多数书的目录集中在前 10-20 页")
    chunk_size = st.slider("文本块大小", min_value=500, max_value=2000, value=1000, step=100)
    retrieve_k = st.slider("检索块数量 (k)", min_value=4, max_value=30, value=15,
                           help="越大覆盖越全，但可能引入噪声")


# ========== 核心函数 ==========

@st.cache_resource
def load_embedding_model():
    """加载 Embedding 模型（只初始化一次）"""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    """复用 LLM 实例，避免每次调用都重新创建"""
    if st.session_state.llm is None:
        st.session_state.llm = ChatOpenAI(
            model=model_name, api_key=api_key, base_url=base_url
        )
    return st.session_state.llm


def process_pdf(file, toc_pages_count: int, chunk_sz: int):
    """
    改进版 PDF 处理：
    - 按页切片，每个 chunk 携带 page 元数据
    - 目录文本带页码锚点，方便 AI 输出页码范围
    """
    doc = fitz.open(stream=file.read(), filetype="pdf")
    toc_text = ""
    all_chunks = []
    all_metadatas = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_sz, chunk_overlap=150
    )
    embeddings = load_embedding_model()

    for i, page in enumerate(doc):
        content = page.get_text()

        # 前 N 页用于提取目录，加上页码锚点
        if i < toc_pages_count:
            toc_text += f"\n--- 第{i + 1}页 ---\n{content}"

        # 按页切块，metadata 记录页码
        chunks = splitter.split_text(content)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"page": i + 1})

    vector_db = FAISS.from_texts(all_chunks, embeddings, metadatas=all_metadatas)
    return vector_db, toc_text, len(doc)


def retrieve_by_page_range(vector_db, query: str, k: int, page_start: int = 0, page_end: int = 0):
    """
    双重保障检索：
    - 如果有页码范围 → metadata 过滤 + 语义搜索
    - 如果没有 → 纯语义搜索
    返回带页码标注的上下文字符串
    """
    if page_start > 0 and page_end >= page_start:
        # FAISS langchain 的 filter 参数支持 dict 形式的简单过滤
        # 但对范围过滤支持有限，所以先取大量候选再手动过滤
        candidates = vector_db.similarity_search(query, k=k * 3)
        docs = [
            d for d in candidates
            if page_start <= d.metadata.get("page", 0) <= page_end
        ][:k]
        # 如果过滤后结果太少，补充纯语义结果
        if len(docs) < k // 2:
            docs = vector_db.similarity_search(query, k=k)
    else:
        docs = vector_db.similarity_search(query, k=k)

    # 按页码排序，保持阅读顺序
    docs.sort(key=lambda d: d.metadata.get("page", 0))

    context = "\n".join(
        [f"[p.{d.metadata.get('page', '?')}] {d.page_content}" for d in docs]
    )
    return context


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
以下是我从原书中按页码检索出的参考片段（方括号内为页码）。

【任务要求】：
1. 严格围绕【{query}】组织内容，不要混入其他章节
2. 提取该部分出现的所有重要专业名词，给出精准定义
3. 按知识框架结构生成内容，层次分明
4. 绝对忠于原文，如果原文没提到的内容，请直接说明"原文未涉及"

【参考原文片段】：
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
                db, toc_text, total_pages = process_pdf(
                    uploaded_file, toc_pages, chunk_size
                )
                st.session_state.vector_db = db

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
        st.info("已扫描全书结构。复制章节名称（含页码）到右侧可获得更精准的结果。")
        st.markdown(st.session_state.framework)

    with col2:
        st.subheader("🎯 按需深度提取")

        chapter_query = st.text_input(
            "输入想学习的部分（建议带页码，例如：第3章 运动学 p.45-p.78）："
        )

        # --- 练习题开关 ---
        generate_exercises = st.checkbox("📝 同时生成练习题", value=False)

        if st.button("🚀 开始提取"):
            if not chapter_query:
                st.warning("请先输入章节名称！")
            else:
                with st.spinner(f"正在精准检索【{chapter_query}】..."):
                    try:
                        llm = get_llm()

                        # 尝试从用户输入中解析页码范围
                        page_start, page_end = 0, 0
                        import re
                        page_match = re.findall(r"p\.?\s*(\d+)", chapter_query)
                        if len(page_match) >= 2:
                            page_start = int(page_match[0])
                            page_end = int(page_match[1])
                        elif len(page_match) == 1:
                            page_start = int(page_match[0])
                            page_end = page_start + 20  # 默认扩展 20 页

                        # 检索
                        context = retrieve_by_page_range(
                            st.session_state.vector_db,
                            chapter_query,
                            retrieve_k,
                            page_start,
                            page_end,
                        )

                        # 生成知识提取
                        detail_result = llm.invoke(
                            PROMPT_DETAIL.format(query=chapter_query, context=context)
                        )
                        st.success(f"✅ 【{chapter_query}】提取完成！")
                        st.markdown(detail_result.content)

                        # 可选：生成练习题
                        if generate_exercises:
                            st.divider()
                            with st.spinner("📝 正在生成练习题..."):
                                exercise_result = llm.invoke(
                                    PROMPT_EXERCISE.format(
                                        query=chapter_query,
                                        content=detail_result.content,
                                    )
                                )
                                st.subheader("📝 练习题")
                                st.markdown(exercise_result.content)

                    except Exception as e:
                        st.error(f"提取出错：{e}")
