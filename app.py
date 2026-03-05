import streamlit as st
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
# 注意：我们彻底删掉了那个容易报错的 RetrievalQA 模块！

# --- 页面设置 ---
st.set_page_config(page_title="AI 全书深度解析器", layout="wide")
st.title("📖 知识点全量提取助手 (DeepSeek 专版)")
st.caption("上传 PDF，自动提取完整框架。基于 DeepSeek 大模型与本地免费向量引擎。")

# --- 初始化 Session State ---
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "framework" not in st.session_state:
    st.session_state.framework = ""

# --- 侧边栏：配置参数 ---
with st.sidebar:
    st.header("1. 配置中心")
    # 尝试从 Streamlit 保险箱读取你的专属 Key，如果没找到，再显示输入框
    try:
        api_key = st.secrets["my_deepseek_key"]
        st.success("✅ 已自动加载内部测试 API Key")
    except:
        api_key = st.text_input("输入 DeepSeek API Key", type="password")
        
    base_url = st.text_input("API 代理地址 (默认 DeepSeek)", value="https://api.deepseek.com")
    model_name = st.selectbox("选择模型", ["deepseek-chat", "deepseek-reasoner"])
    st.info("提示：首次运行本地向量模型会占用几秒钟下载，后续即可秒开。")

# --- 核心功能函数 ---
@st.cache_resource # 缓存向量模型，避免重复加载
def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def process_pdf(file):
    # 1. 解析 PDF
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    
    # 2. 文本切片
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_text(text)
    
    # 3. 建立向量数据库
    embeddings = load_embedding_model()
    vector_db = FAISS.from_texts(chunks, embeddings)
    
    return vector_db, text[:10000] 

# --- UI 交互界面 ---
uploaded_file = st.file_uploader("上传书籍 PDF (最大支持 50MB)", type="pdf")

if uploaded_file and api_key:
    if st.session_state.vector_db is None:
        with st.spinner("正在深度扫描全书内容并构建索引，请稍候..."):
            try:
                db, preview_text = process_pdf(uploaded_file)
                st.session_state.vector_db = db
                
                # 生成初始框架
                llm = ChatOpenAI(
                    model=model_name, 
                    api_key=api_key, 
                    base_url=base_url,
                    max_tokens=4000
                )
                prompt = f"""你是一位拥有20年经验的资深教研专家。请仔细阅读以下文本片段，为你提炼一份系统化的学习指南。

【严格遵守的规则】：
1. 绝对忠于原文，拒绝任何省略，严格按【输出格式模板】输出 Markdown 格式。

【任务要求】：
1. 知识点框架：提取文本中涵盖的所有核心层级结构。
2. 准确定义：为框架中的重要专业名词提供精准、易懂的定义。
3. 实战练习：针对每一个核心概念，设计 2 道练习题（1道基础，1道思考）。

【输出格式模板】：
# 📚 全书核心知识框架
## [章节名称]
### 核心知识点：[知识点名称]
* **📌 准确定义**：[提取的定义]
* **📝 配套练习**：...

文本片段如下：
{preview_text}
"""
                st.session_state.framework = llm.invoke(prompt).content
                st.rerun()
            except Exception as e:
                st.error(f"处理出错啦，请检查代码或网络：{e}")

# --- 结果展示与提问区 ---
if st.session_state.framework:
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("📋 全书知识框架与习题")
        st.markdown(st.session_state.framework)
        
    with col2:
        st.subheader("💬 细节追问")
        user_query = st.text_input("输入你想深入了解的概念：")
        
        if user_query:
            with st.spinner("正在全书中精准检索原文..."):
                try:
                    # 💥 重点修改：弃用容易报错的旧模块，改为最稳定直接的 3 步查询法 💥
                    
                    # 1. 检索：直接从全书切片中找出最相关的 3 段原文
                    docs = st.session_state.vector_db.similarity_search(user_query, k=3)
                    context = "\n".join([doc.page_content for doc in docs])
                    
                    # 2. 组装问题：把原文和用户问题一起打包
                    ask_prompt = f"请作为一位教学助手，基于以下【参考原文】回答问题。如果原文没提到，请结合你的知识解答，但要说明原文未提及。\n\n【参考原文】\n{context}\n\n【用户问题】\n{user_query}"
                    
                    # 3. 发送给 DeepSeek 并输出结果
                    llm = ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)
                    answer = llm.invoke(ask_prompt).content
                    st.info(f"**DeepSeek 深度解答：**\n\n{answer}")
                    
                except Exception as e:
                    st.error(f"检索出错啦：{e}")
