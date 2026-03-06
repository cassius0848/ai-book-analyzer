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
#  页面 & 黑白极简样式
# ================================================================
st.set_page_config(page_title="知识提取器", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&family=JetBrains+Mono:wght@400&display=swap');
html, body, [class*="css"] {
    font-family: 'Noto Sans SC', sans-serif;
    color: #111;
}

.kp-card {
    border-left: 3px solid;
    padding: 14px 18px;
    margin-bottom: 10px;
    background: #fff;
    border-radius: 0;
}
.kp-card.t-S { border-left-color: #111; }
.kp-card.t-A { border-left-color: #444; }
.kp-card.t-B { border-left-color: #888; }
.kp-card.t-C { border-left-color: #bbb; }
.kp-card.t-D { border-left-color: #ddd; }

.kp-head { display:flex; align-items:baseline; gap:8px; margin-bottom:4px; }
.kp-tier {
    font-family:'JetBrains Mono',monospace;
    font-size:0.65em; font-weight:700;
    padding:1px 5px; border-radius:1px;
    letter-spacing:0.5px;
}
.t-S .kp-tier { background:#111; color:#fff; }
.t-A .kp-tier { background:#444; color:#fff; }
.t-B .kp-tier { background:#888; color:#fff; }
.t-C .kp-tier { background:#bbb; color:#fff; }
.t-D .kp-tier { background:#ddd; color:#666; }

.kp-name { font-size:0.92em; font-weight:600; color:#111; }
.kp-def { font-size:0.83em; color:#555; line-height:1.7; margin:4px 0 8px 0; }
.kp-tags { display:flex; gap:5px; flex-wrap:wrap; }
.kp-t {
    font-family:'JetBrains Mono',monospace;
    font-size:0.65em; padding:2px 7px;
    border:1px solid #ddd; color:#666; background:#fafafa;
}

.stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0; }
.stat {
    text-align:center; padding:12px 8px;
    border:1px solid #ddd; background:#fff;
}
.stat-n { font-size:1.6em; font-weight:700; color:#111; }
.stat-l { font-size:0.72em; color:#999; margin-top:2px; }

.lgd { display:flex; gap:14px; flex-wrap:wrap; padding:4px 0; font-size:0.72em; color:#999; }
.lgd-i { display:flex; align-items:center; gap:3px; }
.lgd-d { width:8px; height:8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📖 知识提取器")
st.caption("上传 PDF · 生成目录 · 按章提取 · 权重分析")

# ================================================================
#  Session State
# ================================================================
for k, v in {
    "vector_db":None,"page_texts":{},"framework":"",
    "llm":None,"processing_done":False,
    "knowledge_points":None,"raw_detail":"",
    "mastered":{},"exercises":"","current_query":"",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================================================
#  侧边栏
# ================================================================
with st.sidebar:
    st.markdown("### ⚙️ 配置")
    try:
        api_key = st.secrets["my_deepseek_key"]
        st.success("API 已就绪")
    except Exception:
        api_key = st.text_input("DeepSeek API Key", type="password")
    base_url = st.text_input("API 地址", value="https://api.deepseek.com")
    model_name = st.selectbox("模型", ["deepseek-chat","deepseek-reasoner"])
    st.markdown("---")
    st.markdown("### 参数")
    toc_pages = st.slider("目录扫描页数", 5, 30, 15)
    chunk_size = st.slider("文本块大小", 500, 2000, 1000, step=100)
    retrieve_k = st.slider("语义检索数", 4, 30, 15)
    max_chars = st.slider("单批最大字符", 4000, 30000, 12000, step=2000)

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
        if i < toc_n:
            toc += f"\n--- 第{pn}页 ---\n{t}"
        for c in sp.split_text(t):
            chunks.append(c); metas.append({"page": pn})
    vdb = FAISS.from_texts(chunks, emb, metadatas=metas)
    return vdb, pts, toc, len(doc)

def get_pages(s, e):
    pt = st.session_state.page_texts
    return "\n\n".join([f"[第{p}页]\n{pt[p]}" for p in range(s,e+1) if p in pt])

def sem_search(q, k):
    ds = st.session_state.vector_db.similarity_search(q, k=k)
    ds.sort(key=lambda d: d.metadata.get("page",0))
    return "\n".join([f"[p.{d.metadata.get('page','?')}] {d.page_content}" for d in ds])

def parse_pr(text):
    m = re.findall(r"p\.?\s*(\d+)", text)
    if len(m)>=2: return int(m[0]),int(m[1])
    if len(m)==1: return int(m[0]),int(m[0])+20
    m = re.findall(r"第\s*(\d+)\s*页", text)
    if len(m)>=2: return int(m[0]),int(m[1])
    if len(m)==1: return int(m[0]),int(m[0])+20
    return 0,0

# ================================================================
#  JSON 解析（多层容错）
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
#  提取 & Map-Reduce
# ================================================================
EXTRACT_PROMPT = """你是一位严谨的教研专家。用户是初学者，想深度学习【{query}】。

以下是原文{batch_note}（方括号内为页码）。

■ 语言要求：
- 整体中文撰写
- 专业术语用「中文名 (English Term)」格式
- 定义用中文

■ 权重（1-5）：
- core：核心度
- difficulty：难度
- connectivity：关联度
- exam_weight：考试权重

■ 输出：只输出 JSON 数组。直接 [ 开头 ] 结尾，不要任何其他文字。

每个元素：
{{"id":"KP-01","name":"稳定匹配 (Stable Matching)","category":"所属主题","definition":"中文精准定义2-4句","page":"p.23","weights":{{"core":4,"difficulty":3,"connectivity":5,"exam_weight":4}},"prerequisites":["KP-XX"],"related":["KP-XX"]}}

原文：
{text}"""

def extract_single(text, query, llm, batch_label=None):
    bn = f"（第{batch_label}部分）" if batch_label else ""
    resp = llm.invoke(EXTRACT_PROMPT.format(query=query, batch_note=bn, text=text)).content
    p = robust_json(resp)
    return p if p else resp

def batch_extract(full_text, query, llm, mc):
    if len(full_text) <= mc:
        return extract_single(full_text, query, llm)
    pages = full_text.split("\n\n")
    batches, cur = [], ""
    for b in pages:
        if len(cur)+len(b)>mc and cur:
            batches.append(cur); cur=b
        else:
            cur += ("\n\n"+b) if cur else b
    if cur: batches.append(cur)
    all_kps = []
    prog = st.progress(0, text="提取中...")
    for i, batch in enumerate(batches):
        prog.progress((i+1)/(len(batches)+1), text=f"第 {i+1}/{len(batches)} 批")
        r = extract_single(batch, query, llm, f"{i+1}/{len(batches)}")
        if isinstance(r, list): all_kps.extend(r)
    prog.progress(1.0, text="合并中...")
    if not all_kps:
        prog.empty(); return "未返回有效数据"
    rp = f"""合并以下知识点，去重。只输出 JSON 数组，直接 [ 开头 ] 结尾。
保持中文，专业名词「中文 (English)」。
{json.dumps(all_kps, ensure_ascii=False)}"""
    resp = llm.invoke(rp).content
    prog.empty()
    p = robust_json(resp)
    return p if p else all_kps

# ================================================================
#  可视化（黑白灰配色）
# ================================================================
def tier(c):
    if c>=5: return "S"
    if c>=4: return "A"
    if c>=3: return "B"
    if c>=2: return "C"
    return "D"

TIER_GRAY = {"S":"#111","A":"#444","B":"#888","C":"#bbb","D":"#ddd"}

def card_html(kp):
    w = kp.get("weights",{})
    c = w.get("core",3)
    t = tier(c)
    return f"""<div class="kp-card t-{t}">
<div class="kp-head"><span class="kp-tier">{t}</span><span class="kp-name">{kp.get('id','')}  {kp.get('name','')}</span></div>
<div class="kp-def">{kp.get('definition','—')}</div>
<div class="kp-tags">
<span class="kp-t">核心 {c}/5</span>
<span class="kp-t">难度 {w.get('difficulty','?')}/5</span>
<span class="kp-t">关联 {w.get('connectivity','?')}/5</span>
<span class="kp-t">考试 {w.get('exam_weight','?')}/5</span>
<span class="kp-t">{kp.get('page','')}</span>
</div></div>"""

def make_treemap(kps):
    names,parents,vals,cols,hvs = [],[],[],[],[]
    cats = {}
    for kp in kps:
        cats.setdefault(kp.get("category","未分类"),[]).append(kp)
    root = "全部"
    names.append(root); parents.append(""); vals.append(0); cols.append(0); hvs.append("")
    for cat, items in cats.items():
        names.append(cat); parents.append(root); vals.append(0); cols.append(0)
        hvs.append(f"{cat} ({len(items)})")
        for kp in items:
            w = kp.get("weights",{})
            s = sum(w.get(k,3) for k in ["core","difficulty","connectivity","exam_weight"])
            names.append(kp["name"]); parents.append(cat)
            vals.append(s); cols.append(w.get("core",3))
            hvs.append(f"{kp['name']}\n核心{w.get('core',0)} 难度{w.get('difficulty',0)} 关联{w.get('connectivity',0)} 考试{w.get('exam_weight',0)}")
    fig = go.Figure(go.Treemap(
        labels=names, parents=parents, values=vals,
        marker=dict(colors=cols,
                    colorscale=[[0,"#f5f5f5"],[0.3,"#ccc"],[0.6,"#777"],[1,"#111"]],
                    line=dict(width=1.5, color="#fff")),
        hovertext=hvs, hoverinfo="text", textinfo="label",
        textfont=dict(size=12, family="Noto Sans SC"),
        pathbar=dict(visible=True)))
    fig.update_layout(margin=dict(t=20,l=6,r=6,b=6), height=420,
                      font=dict(family="Noto Sans SC"), paper_bgcolor="#fff")
    return fig

def make_radar(kp):
    w = kp.get("weights",{})
    cs = ["核心度","难度","关联度","考试权重"]
    vs = [w.get("core",0),w.get("difficulty",0),w.get("connectivity",0),w.get("exam_weight",0)]
    vs.append(vs[0])
    fig = go.Figure(go.Scatterpolar(
        r=vs, theta=cs+[cs[0]], fill="toself",
        fillcolor="rgba(0,0,0,0.04)", line=dict(color="#333",width=1.5),
        marker=dict(size=5,color="#333")))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True,range=[0,5],tickvals=[1,2,3,4,5],
                                   tickfont=dict(size=9,color="#aaa"),gridcolor="#e8e8e8"),
                   angularaxis=dict(tickfont=dict(size=10,color="#666")),
                   bgcolor="#fff"),
        showlegend=False, margin=dict(t=12,b=12,l=40,r=40), height=220,
        font=dict(family="Noto Sans SC"), paper_bgcolor="#fff")
    return fig

def make_bubble(kps):
    data = []
    for kp in kps:
        w = kp.get("weights",{})
        data.append({"知识点":kp["name"],"核心度":w.get("core",3),"难度":w.get("difficulty",3),
                     "考试权重":w.get("exam_weight",3),
                     "综合":w.get("core",3)+w.get("exam_weight",3)})
    fig = px.scatter(data, x="核心度",y="难度",size="综合",color="考试权重",
                     hover_name="知识点",size_max=28,
                     color_continuous_scale=[[0,"#ddd"],[0.5,"#888"],[1,"#111"]])
    fig.update_layout(height=360,margin=dict(t=20,b=20),
                      font=dict(family="Noto Sans SC"),paper_bgcolor="#fff",plot_bgcolor="#fafafa",
                      xaxis=dict(range=[0.5,5.5],dtick=1,gridcolor="#eee"),
                      yaxis=dict(range=[0.5,5.5],dtick=1,gridcolor="#eee"))
    return fig

# ================================================================
#  Prompt
# ================================================================
PROMPT_TOC = """你是图书目录整理专家。以下是书籍前若干页内容（已标注页码）。

■ 语言：章节标题翻译为中文，英文原标题放括号中。
■ 格式：每行一个条目，章前空行，子节缩进2空格。

示例：

第1章 引言 (Introduction) (p.1 - p.28)
  1.1 稳定匹配问题 (Stable Matching) (p.1 - p.12)
  1.2 五个代表性问题 (Five Representative Problems) (p.12 - p.19)

第2章 算法分析基础 (Basics of Algorithm Analysis) (p.29 - p.70)
  2.1 计算可处理性 (Computational Tractability) (p.29 - p.35)

要求：每条标注页码，不省略，只输出目录。

文本：
{toc_text}"""

PROMPT_EX = """基于以下知识点，针对【{query}】出题。中文出题，专业名词标英文。

基础题（1道）：核心定义
思考题（1道）：综合运用

格式：
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
uploaded = st.file_uploader("上传 PDF", type="pdf")

if uploaded and api_key:
    if not st.session_state.processing_done:
        with st.spinner("处理中..."):
            try:
                db, pts, toc, n = process_pdf(uploaded, toc_pages, chunk_size)
                st.session_state.vector_db = db
                st.session_state.page_texts = pts
                llm = get_llm()
                r = llm.invoke(PROMPT_TOC.format(toc_text=toc))
                st.session_state.framework = r.content
                st.session_state.processing_done = True
                st.rerun()
            except Exception as e:
                st.error(f"出错：{e}")
elif not api_key:
    st.info("请先配置 API Key")

# ================================================================
#  目录 + 提取
# ================================================================
if st.session_state.framework:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("#### 📑 目录")
        st.caption("复制含页码条目到右侧")
        st.markdown(st.session_state.framework)
    with c2:
        st.markdown("#### 🔍 提取")
        query = st.text_input("输入章节（含页码）", placeholder="1.1 稳定匹配 p.1-p.12")
        gen_ex = st.checkbox("生成练习题", value=False)
        if st.button("开始提取", use_container_width=True):
            if not query:
                st.warning("请输入章节")
            else:
                try:
                    llm = get_llm()
                    ps, pe = parse_pr(query)
                    if ps > 0:
                        st.caption(f"精确模式 · p.{ps}–{pe}")
                        ctx = get_pages(ps, pe)
                        if not ctx.strip():
                            st.error("页码范围无内容"); st.stop()
                        result = batch_extract(ctx, query, llm, max_chars)
                    else:
                        st.caption("语义搜索模式")
                        ctx = sem_search(query, retrieve_k)
                        result = extract_single(ctx, query, llm)
                    if isinstance(result, list):
                        st.session_state.knowledge_points = result
                        st.session_state.raw_detail = json.dumps(result, ensure_ascii=False, indent=2)
                    else:
                        st.session_state.knowledge_points = None
                        st.session_state.raw_detail = result if isinstance(result,str) else str(result)
                    st.session_state.current_query = query
                    if gen_ex:
                        with st.spinner("生成练习题..."):
                            c = json.dumps(result,ensure_ascii=False) if isinstance(result,list) else str(result)
                            st.session_state.exercises = llm.invoke(
                                PROMPT_EX.format(query=query, content=c)).content
                    st.rerun()
                except Exception as e:
                    st.error(f"出错：{e}")

# ================================================================
#  结果展示
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
    st.markdown(f"""<div class="stats">
<div class="stat"><div class="stat-n">{total}</div><div class="stat-l">知识点</div></div>
<div class="stat"><div class="stat-n">{hi}</div><div class="stat-l">核心</div></div>
<div class="stat"><div class="stat-n">{ad:.1f}</div><div class="stat-l">平均难度</div></div>
<div class="stat"><div class="stat-n">{mc}/{total}</div><div class="stat-l">已掌握</div></div>
</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="lgd">
<div class="lgd-i"><div class="lgd-d" style="background:#111"></div>S 必须掌握</div>
<div class="lgd-i"><div class="lgd-d" style="background:#444"></div>A 重要</div>
<div class="lgd-i"><div class="lgd-d" style="background:#888"></div>B 一般</div>
<div class="lgd-i"><div class="lgd-d" style="background:#bbb"></div>C 了解</div>
<div class="lgd-i"><div class="lgd-d" style="background:#ddd"></div>D 补充</div>
</div>""", unsafe_allow_html=True)

    t1,t2,t3,t4 = st.tabs(["图谱","卡片","矩阵","数据"])

    with t1:
        st.plotly_chart(make_treemap(kps), use_container_width=True)
        st.caption("面积 = 综合权重 · 颜色深浅 = 核心度")

    with t2:
        fc1,fc2,fc3 = st.columns(3)
        with fc1: fcore = st.select_slider("最低核心度",[1,2,3,4,5],value=1)
        with fc2: fsort = st.selectbox("排序",["核心度","难度","考试权重","页码"])
        with fc3: fstat = st.selectbox("状态",["全部","未掌握","已掌握"])

        fl = [k for k in kps if k.get("weights",{}).get("core",0)>=fcore]
        if fstat=="未掌握":
            fl=[k for k in fl if not st.session_state.mastered.get(k.get("id",""),False)]
        elif fstat=="已掌握":
            fl=[k for k in fl if st.session_state.mastered.get(k.get("id",""),False)]
        sm = {
            "核心度":lambda k:-k.get("weights",{}).get("core",0),
            "难度":lambda k:-k.get("weights",{}).get("difficulty",0),
            "考试权重":lambda k:-k.get("weights",{}).get("exam_weight",0),
            "页码":lambda k:int(re.search(r"\d+",k.get("page","0")).group()) if re.search(r"\d+",k.get("page","0")) else 0,
        }
        fl.sort(key=sm.get(fsort,sm["核心度"]))
        st.caption(f"{len(fl)}/{len(kps)}")

        for i, kp in enumerate(fl):
            cc1,cc2 = st.columns([7,1])
            with cc1: st.markdown(card_html(kp), unsafe_allow_html=True)
            with cc2:
                kid = kp.get("id",f"k{i}")
                mv = st.checkbox("✓",value=st.session_state.mastered.get(kid,False),
                                 key=f"m_{kid}",help="已掌握")
                st.session_state.mastered[kid] = mv

            with st.expander(f"{kp['name']}"):
                rc1,rc2 = st.columns([1,1])
                with rc1: st.plotly_chart(make_radar(kp), use_container_width=True)
                with rc2:
                    w = kp.get("weights",{})
                    tot = sum(w.values())
                    t_name = tier(w.get("core",0))
                    st.markdown(f"**{t_name}级 · {tot}/20**")
                    st.markdown(f"""
| 维度 | 值 | 说明 |
|---|---|---|
| 核心度 | {w.get('core',0)}/5 | {"必须掌握" if w.get('core',0)>=4 else "建议了解" if w.get('core',0)>=3 else "可选"} |
| 难度 | {w.get('difficulty',0)}/5 | {"重点攻克" if w.get('difficulty',0)>=4 else "中等" if w.get('difficulty',0)>=3 else "容易"} |
| 关联度 | {w.get('connectivity',0)}/5 | {"枢纽" if w.get('connectivity',0)>=4 else "有关联" if w.get('connectivity',0)>=3 else "独立"} |
| 考试 | {w.get('exam_weight',0)}/5 | {"高频" if w.get('exam_weight',0)>=4 else "可能" if w.get('exam_weight',0)>=3 else "低频"} |
""")
                    pre = kp.get("prerequisites",[])
                    rel = kp.get("related",[])
                    if pre: st.caption(f"前置：{', '.join(pre)}")
                    if rel: st.caption(f"相关：{', '.join(rel)}")

    with t3:
        st.plotly_chart(make_bubble(kps), use_container_width=True)
        st.caption("X = 核心度 · Y = 难度 · 大小 = 综合 · 深浅 = 考试权重")

    with t4:
        st.code(st.session_state.raw_detail, language="json")

    if st.session_state.exercises:
        st.markdown("---")
        st.markdown("#### ✏️ 练习题")
        st.markdown(st.session_state.exercises)

elif st.session_state.raw_detail and not st.session_state.knowledge_points:
    st.markdown("---")
    st.markdown(f"#### {st.session_state.current_query}")
    st.caption("结构化解析未成功，文本模式")
    raw = st.session_state.raw_detail
    last = robust_json(raw)
    if last:
        st.session_state.knowledge_points = last; st.rerun()
    else:
        st.markdown(raw)
    if st.session_state.exercises:
        st.markdown("---")
        st.markdown("#### ✏️ 练习题")
        st.markdown(st.session_state.exercises)
