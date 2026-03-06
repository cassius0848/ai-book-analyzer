import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

# --- 页面设置 ---
st.set_page_config(page_title="AI 文件整理解析器", layout="wide")
st.title("📖 知识点全量提取助手 (大文件架构)")
st.caption("采用“分而治之”架构：左侧生成全局目录，右侧按需提取详细定义与习题")

# --- 初始化 Session State ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "framework" not in st.session_state:
    st.session_state.framework = ""

# --- 侧边栏：配置参数 ---
with st.sidebar:
    st.header("1. 配置中心")
    try:
        api_key = st.secrets["my_deepseek_key"]
        st.success("✅ 已自动加载内部测试 API Key")
    except:
        api_key = st.text_input("输入 DeepSeek API Key", type="password")
        
    base_url = st.text_input("API 代理地址", value="https://api.deepseek.com")
    model_name = st.selectbox("选择模型", ["deepseek-chat", "deepseek-reasoner"])

# --- 核心功能函数 ---
@st.cache_resource 
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf_map_reduce(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    toc_text = "" # 专门用来装前 15 页的内容，用于提取目录
    
    for i, page in enumerate(doc):
        page_content = page.get_text()
        full_text += page_content
        if i < 15: # 大多数书的目录都在前 15 页
            toc_text += page_content
            
    # 把整本书切成小块存入数据库 (为右侧的详细提取做准备)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(full_text)
    
    embeddings = load_embedding_model()
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    return vector_db, toc_text

# --- UI 交互界面 ---
uploaded_file = st.file_uploader("上传书籍 PDF (支持 30MB+ 大文件)", type="pdf")

if uploaded_file and api_key:
    if st.session_state.vector_db is None:
        with st.spinner("第一步：正在切片整本书并提取全局目录，请稍候..."):
            try:
                db, toc_text = process_pdf_map_reduce(uploaded_file)
                st.session_state.vector_db = db
                
                # 第一步 (Map): 只让 AI 读取前 15 页，生成全局知识树
                llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
                prompt_toc = f"""你是一个图书目录整理专家。请阅读以下书籍开头的文本，只需要整理目录，提取出这本的【全局目录框架/知识树】。
只输出章节的层级结构（例如：第一章 XX，1.1 XX），不要输出任何正文解释，不要省略任何章节。
文本：{toc_text}"""
                
                st.session_state.framework = llm.invoke(prompt_toc).content
                st.rerun()
            except Exception as e:
                st.error(f"处理出错啦：{e}")

# --- 结果展示与提问区 (分而治之) ---
if st.session_state.framework:
    col1, col2 = st.columns([1, 1.5]) # 调整左右比例
    
    with col1:
        st.subheader("🌳 第一步：全局目录树")
        st.info("已扫描全书结构。请查看下方目录，并在右侧输入你想深入的章节。")
        st.markdown(st.session_state.framework)
        
    with col2:
        st.subheader("🎯 第二步：按需深度提取 (点读机)")
        chapter_query = st.text_input("请输入你想详细学习的部分（例如：第一章、或者某个具体小节）：")
        
        if st.button("🚀 生成该部分的详细定义与习题"):
            if chapter_query:
                with st.spinner(f"正在全书中精准抓取【{chapter_query}】的所有细节，这可能需要十几秒..."):
                    try:
                        # 1. 检索：针对这一章，从几百页的书里抓取最相关的 8 个大文本块
                        docs = st.session_state.vector_db.similarity_search(chapter_query, k=8)
                        context = "\n".join([doc.page_content for doc in docs])
                        
                        # 2. 第二步 (Reduce): 专注处理这一章，绝不省略！
                        prompt_detail = f"""你是一位严谨的教研专家。用户现在是初学者，想深度学习【{chapter_query}】。
请基于以下我为你从原书中找出的【参考原文片段】，生成一份毫无遗漏的学习框架结构。

【任务要求】：
1. 在整理好的目录中找出对应章节中用户想寻找的知识内容
2. 准确定义：提取这段原文中出现的所有重要专业名词，并给出精准定义。
3. 按照知识框架结构生成内容
4. 实战练习：针对这些核心概念，设计 2 道练习题（1道基础题，1道深度思考题）。
5. 绝对忠于原文，如果原文没提，请直接说明。

【参考原文片段】：
{context}
"""
                        llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
                        answer = llm.invoke(prompt_detail).content
                        
                        st.success(f"✅ 【{chapter_query}】深度提炼完成！")
                        st.markdown(answer)
                        
                    except Exception as e:
                        st.error(f"提取出错啦：{e}")
            else:
                st.warning("请先在上方输入框填写章节名称哦！")

