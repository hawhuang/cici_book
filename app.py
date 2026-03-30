import streamlit as st
import dashscope
from dashscope import MultiModalConversation, Generation, ImageSynthesis
import io, re, os, socket, qrcode, base64, requests, json
import pandas as pd
from PIL import Image

# --- 1. 基础配置 ---
dashscope.api_key = st.secrets["dashscope"]["api_key"]

VISION_MODEL = 'qwen-vl-max'
TEXT_MODEL = 'qwen-max'
IMAGE_MODEL = 'qwen-image-2.0'

# Turso HTTP API 配置
# 将 libsql:// 协议转换为 https:// 用于 HTTP API
_turso_raw_url = st.secrets["turso"]["url"]
TURSO_HTTP_URL = _turso_raw_url.replace("libsql://", "https://")
TURSO_TOKEN = st.secrets["turso"]["token"]

# --- 2. Turso HTTP API 数据库操作 ---
def _turso_execute(statements):
    """通过 Turso HTTP API 执行 SQL 语句
    
    Args:
        statements: SQL 语句列表，每项可以是:
            - 字符串: 简单 SQL
            - dict: {"q": "SQL", "params": [...]} 带参数的 SQL
    Returns:
        API 响应的 JSON（包含 results 数组）
    """
    url = f"{TURSO_HTTP_URL}/v3/pipeline"
    headers = {
        "Authorization": f"Bearer {TURSO_TOKEN}",
        "Content-Type": "application/json"
    }
    
    # 构建 pipeline 请求体
    req_body = {"requests": []}
    for stmt in statements:
        if isinstance(stmt, str):
            req_body["requests"].append({"type": "execute", "stmt": {"sql": stmt}})
        elif isinstance(stmt, dict):
            s = {"sql": stmt["q"]}
            if "params" in stmt:
                # 位置参数
                s["args"] = [{"type": "text", "value": str(v)} for v in stmt["params"]]
            req_body["requests"].append({"type": "execute", "stmt": s})
    # 最后加一个 close
    req_body["requests"].append({"type": "close"})
    
    resp = requests.post(url, headers=headers, json=req_body, timeout=30)
    resp.raise_for_status()
    return resp.json()

def init_db():
    """初始化数据库表结构"""
    _turso_execute([
        """CREATE TABLE IF NOT EXISTS vocabulary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('word', 'char')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(content, type)
        )"""
    ])

def load_history():
    """从 Turso 数据库加载历史记录"""
    all_words, all_chars = set(), set()
    try:
        result = _turso_execute(["SELECT content, type FROM vocabulary"])
        # 解析 pipeline 响应: results[0].response.result.rows
        rows = result.get("results", [{}])[0].get("response", {}).get("result", {}).get("rows", [])
        for row in rows:
            # 每行是一个数组，每个元素是 {"type": "text", "value": "..."}
            content = row[0].get("value", "")
            vtype = row[1].get("value", "")
            if vtype == 'word':
                all_words.add(content)
            elif vtype == 'char':
                all_chars.add(content)
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
    return sorted(list(all_words)), sorted(list(all_chars))

def save_history(words, chars):
    """保存更新后的列表到 Turso 数据库（全量覆盖）"""
    try:
        stmts = ["DELETE FROM vocabulary"]
        for w in words:
            if w.strip():
                stmts.append({"q": "INSERT OR IGNORE INTO vocabulary (content, type) VALUES (?, ?)", "params": [w.strip(), "word"]})
        for c in chars:
            if c.strip():
                stmts.append({"q": "INSERT OR IGNORE INTO vocabulary (content, type) VALUES (?, ?)", "params": [c.strip(), "char"]})
        _turso_execute(stmts)
    except Exception as e:
        st.error(f"保存失败: {str(e)}")

def append_history(words, chars):
    """追加新词到 Turso 数据库（不覆盖已有数据）"""
    try:
        stmts = []
        for w in words:
            if w.strip():
                stmts.append({"q": "INSERT OR IGNORE INTO vocabulary (content, type) VALUES (?, ?)", "params": [w.strip(), "word"]})
        for c in chars:
            if c.strip():
                stmts.append({"q": "INSERT OR IGNORE INTO vocabulary (content, type) VALUES (?, ?)", "params": [c.strip(), "char"]})
        if stmts:
            _turso_execute(stmts)
    except Exception as e:
        st.error(f"追加保存失败: {str(e)}")

def get_image_base64(url):
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return base64.b64encode(response.content).decode()
    except: pass
    return None

def clean_prompt(text):
    """极致净化：只保留汉字、字母和数字，彻底避免 URL 误判"""
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text).strip()

# --- 3. 生成故事函数 ---
def generate_story(recognized_content, language='zh'):
    import random
    
    if language == 'zh':
        roles = ["超级飞侠乐迪", "艾莎公主", "汪汪队", "美人鱼爱丽儿","巴拉巴拉小魔仙", "小猪佩奇","白雪公主"]
        role = random.choice(roles)
        sys_prompt = f"你是一位幼儿园老师，为5岁女孩写温馨简单的童话。故事的主人公是{role}。"
        recognized_str = "、".join(recognized_content)
        user_prompt = f"使用这些词：{recognized_str}。写一个70字左右的故事，分三个场景。"
    else:
        roles = ["Super Wings Jett", "Elsa", "PAW Patrol", "Mermaid Ariel"]
        role = random.choice(roles)
        sys_prompt = f"You are a kindergarten teacher writing warm and simple fairy tales for a 5-year-old girl. The protagonist is {role}."
        recognized_str = ", ".join(recognized_content)
        user_prompt = f"Use these words: {recognized_str}. Write a story of about 70 words in English, divided into three scenes."
    
    try:
        response = Generation.call(
            model=TEXT_MODEL,
            messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': user_prompt}],
            result_format='message'
        )
        return response.output.choices[0].message.content if response.status_code == 200 else None
    except: return None

# --- 4. 终极修复版绘图函数 ---
def generate_images(story_text):
    """采用显式 input 字典调用，彻底杜绝 URL Error"""
    # 0. 提取主角特征以保持人物一致性
    try:
        char_res = Generation.call(
            model=TEXT_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are an assistant who extracts visual descriptions.'},
                {'role': 'user', 'content': f"Read this story: {story_text}\n\nDescribe the main character's appearance in 10 English words for image generation (e.g., 'cute little girl with long hair', 'blue airplane robot'). If not specified, invent a consistent cute look."}
            ],
            result_format='message'
        )
        char_desc = char_res.output.choices[0].message.content
    except:
        char_desc = "cute 5 year old girl" # Fallback

    # 提取三个干净的视觉场景描述
    segments = [story_text[:40], story_text[len(story_text)//3:len(story_text)//3+40], story_text[-40:]]
    image_data_list = []
    st.session_state.debug_api_log = []
    
    # Generate a random seed for style consistency
    import random
    fixed_seed = random.randint(1, 10000)
    
    for idx, seg in enumerate(segments):
        # 这里的关键词是防止识别为 URL 的关键
        visual_prompt = f"Pixar style, {char_desc}, {clean_prompt(seg)}"
        
        try:
            # 官方 SDK 对 qwen-image-2.0 的支持可能存在参数映射问题，
            # 导致 'prompt' 参数被错误处理或者没有正确传递 input.messages。
            # 这里直接使用 requests 调用 REST API，绕过 SDK 的潜在封装问题，确保参数结构完全符合文档要求。
            
            headers = {
                "Authorization": f"Bearer {dashscope.api_key}",
                "Content-Type": "application/json"
            }
            # 文档: https://help.aliyun.com/zh/model-studio/qwen-image-api
            # qwen-image-2.0 是同步接口 (不需要 task_id 轮询)，直接返回 output.results
            
            # 但为了避免 SDK 的 url error，直接构造最纯净的 payload
            payload = {
                "model": IMAGE_MODEL,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": visual_prompt}]
                        }
                    ]
                },
                "parameters": {
                    "size": "1024*1024",
                    "n": 1,
                    "seed": fixed_seed
                }
            }
            
            # 使用 requests.post 调用
            import requests # 确保已导入
            response = requests.post(
                "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                headers=headers,
                json=payload,
                timeout=60 # 设置较长超时，因为是同步生成
            )
            
            st.session_state.debug_api_log.append(f"场景 {idx+1} 状态码: {response.status_code}")
            
            if response.status_code == 200:
                res_json = response.json()
                img_url = None
                
                # 尝试解析不同的响应结构
                if 'output' in res_json:
                    output = res_json['output']
                    # 结构 A: output.results[0].url (旧版或部分模型)
                    if 'results' in output and isinstance(output['results'], list):
                        img_url = output['results'][0].get('url')
                    # 结构 B: output.choices[0].message.content[0].image (新版 Qwen-Image-2.0)
                    elif 'choices' in output and isinstance(output['choices'], list):
                        content = output['choices'][0]['message']['content']
                        if isinstance(content, list) and 'image' in content[0]:
                             img_url = content[0]['image']
                
                if img_url:
                    b64 = get_image_base64(img_url)
                    if b64: image_data_list.append(b64)
                else:
                    st.session_state.debug_api_log.append(f"  ❌ 响应解析失败: {res_json}")
            else:
                 st.session_state.debug_api_log.append(f"  ❌ API 报错: {response.text}")

        except Exception as e:
            st.session_state.debug_api_log.append(f"  ❌ 运行时异常: {str(e)}")
            
    return image_data_list

# --- 5. UI 逻辑 ---
# 初始化数据库（仅首次运行时创建表）
init_db()

st.set_page_config(page_title="西西的识字助手", layout="centered", page_icon="🌈")

st.markdown("""<style>
    /* 强制亮色背景，覆盖暗黑模式 */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
    .main, .block-container, section[data-testid="stSidebar"] {
        background: linear-gradient(160deg, #FFF6F0 0%, #F0F4FF 40%, #FFF0F8 70%, #F0FFF4 100%) !important;
        color: #3D2C50 !important;
        font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
    }
    /* 全面覆盖文字颜色 */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp div,
    .stApp .stMarkdown, .stApp [data-testid="stText"],
    .stApp [data-testid="stCaptionContainer"],
    .stApp .stAlert p, .stApp small, .stApp code,
    [data-testid="stExpander"] p, [data-testid="stExpander"] span,
    .stApp [data-baseweb="tab"] { color: #3D2C50 !important; }
    .stApp a { color: #8B5CF6 !important; }
    h1 {
        background: linear-gradient(135deg, #FF6B9D, #C44EFF, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900 !important;
        font-size: 2.4rem !important;
        text-align: center;
    }
    h2, h3 { color: #6C4AB6 !important; font-weight: 700 !important; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #FFFFFF, #F8F0FF) !important;
        padding: 22px; border-radius: 20px;
        border: 2px solid #E8D5F5;
    }
    div[data-testid="stMetric"] label { color: #8B5CF6 !important; font-weight: 700 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #C44EFF !important; font-weight: 900 !important; font-size: 2.2rem !important; }
    .stButton > button {
        background: linear-gradient(135deg, #FF6B9D 0%, #C44EFF 100%) !important;
        color: white !important; border: none !important;
        border-radius: 25px !important; padding: 0.55rem 1.8rem !important;
        font-weight: 700 !important; font-size: 1rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #FF85B1 0%, #D06CFF 100%) !important;
    }
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #4ECDC4 0%, #44B09E 100%) !important;
        color: white !important; border: none !important;
        border-radius: 25px !important; padding: 0.55rem 1.8rem !important;
        font-weight: 700 !important; font-size: 1rem !important;
    }
    details summary { color: #6C4AB6 !important; font-weight: 700 !important; }
    details summary span { color: #6C4AB6 !important; }
    details { border: 2px solid #F0E0FF !important; border-radius: 18px !important; background: #FFFBFE !important; }
    .story-box {
        background: linear-gradient(135deg, #FFF5F9 0%, #F0F4FF 50%, #F5FFF0 100%);
        border: 2.5px solid #E8D5F5; padding: 28px 25px;
        border-radius: 24px; line-height: 2.2; font-size: 1.6rem; color: #4A3660 !important;
    }
    .stTextInput input, .stTextArea textarea {
        color: #3D2C50 !important; background: #FFFFFF !important;
        border-radius: 14px !important; border: 2px solid #E8D5F5 !important;
    }
    .stFileUploader > div { border-radius: 18px !important; border: 2px dashed #D5B8FF !important; background: #FEFAFF !important; }
    .stImage > img { border-radius: 18px !important; }
    .stTabs [data-baseweb="tab"] { color: #8B5CF6 !important; font-weight: 600 !important; }
    /* 强制 expander 内容区亮色 */
    [data-testid="stExpander"] > div { background: #FFFBFE !important; }
    /* 选择框、下拉框等 */
    .stSelectbox [data-baseweb="select"], .stMultiSelect [data-baseweb="select"] {
        background: #FFFFFF !important; color: #3D2C50 !important;
    }
    footer { visibility: hidden; }
</style>""", unsafe_allow_html=True)

if 'story_content' not in st.session_state: st.session_state.story_content = ""
if 'story_imgs' not in st.session_state: st.session_state.story_imgs = []
if 'ocr_words' not in st.session_state: st.session_state.ocr_words = ""
if 'ocr_chars' not in st.session_state: st.session_state.ocr_chars = ""
if 'ocr_done' not in st.session_state: st.session_state.ocr_done = False
if 'princess_count' not in st.session_state: st.session_state.princess_count = 0

st.title("🌈 西西的魔法识字乐园")

# 公主 logo 区域
PRINCESS_EMOJIS = ["👸", "👑", "🏰", "💖", "🦄", "🌸", "✨", "🎀", "💎", "🩰"]

if st.button("👸 召唤公主"):
    st.session_state.princess_count += 1

if st.session_state.princess_count > 0:
    logos = ""
    for i in range(st.session_state.princess_count):
        logos += PRINCESS_EMOJIS[i % len(PRINCESS_EMOJIS)] + " "
    st.markdown(
        f'<div style="text-align:center; font-size:2.2rem; letter-spacing:6px; padding:8px 0;">{logos.strip()}</div>',
        unsafe_allow_html=True
    )
words_list, chars_list = load_history()

c1, c2 = st.columns(2)
with c1: st.metric("🔤 累计单词", len(words_list))
with c2: st.metric("🀄 累计汉字", len(chars_list))

with st.expander("🔍 查询已记录的字词"):
    search_term = st.text_input("输入要查找的汉字或单词", "")
    if search_term:
        found_chars = [c for c in chars_list if search_term in c]
        found_words = [w for w in words_list if search_term.lower() in w.lower()]
        
        if found_chars:
            st.write("📝 **找到的汉字:**")
            st.write(" ".join(found_chars))
        
        if found_words:
            st.write("📝 **找到的单词:**")
            st.write(", ".join(found_words))
            
        if not found_chars and not found_words:
            st.write("🤷‍♂️ 未找到相关记录")
    else:
        st.write("**所有记录预览:**")
        tab1, tab2 = st.tabs(["汉字列表", "单词列表"])
        with tab1:
            st.write(" ".join(chars_list) if chars_list else "暂无记录")
        with tab2:
            st.write(", ".join(words_list) if words_list else "暂无记录")

with st.expander("✏️ 编辑生词本"):
    with st.form("edit_form"):
        st.caption("在这里可以直接修改、添加或删除记录，保存后会覆盖原有数据。")
        
        # 预填充数据
        edit_words = st.text_area("英语单词 (用逗号或空格分隔)", value=", ".join(words_list), height=100)
        edit_chars = st.text_area("汉字 (用空格分隔)", value=" ".join(chars_list), height=100)
        
        if st.form_submit_button("💾 保存修改"):
            # 处理单词：支持逗号、空格、换行分隔
            new_words = [w.strip() for w in re.split(r'[,\s\n]+', edit_words) if w.strip()]
            # 简单去重并排序
            new_words = sorted(list(set(new_words)))
            
            # 处理汉字
            new_chars = [c.strip() for c in re.split(r'[,\s\n]+', edit_chars) if c.strip()]
            new_chars = sorted(list(set(new_chars)))
            
            save_history(new_words, new_chars)
            st.success("✅ 修改已保存！")
            st.rerun()


with st.expander("📷 拍照识字"):
    files = st.file_uploader("上传图片，自动提取文字", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    # 第一步：识别
    if files and st.button("🔍 开始识别"):
        all_words, all_chars = [], []
        progress_bar = st.progress(0, text="准备识别...")
        total = len(files)
        has_error = False
        
        for i, f in enumerate(files):
            progress_bar.progress((i) / total, text=f"正在识别第 {i+1}/{total} 张图片...")
            try:
                # 压缩图片以减小请求体，避免传输超时
                img = Image.open(f)
                # 如果图片过大，缩放到最大 1024px
                max_size = 1024
                if max(img.size) > max_size:
                    ratio = max_size / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                # 转为 JPEG 格式压缩
                buf = io.BytesIO()
                img.convert('RGB').save(buf, format='JPEG', quality=85)
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                
                msg = [{"role": "user", "content": [{"image": f"data:image/jpeg;base64,{b64}"}, {"text": "请提取图片中的所有文字，包括中文和英文"}]}]
                
                # 使用 requests 直接调用 API，可以控制超时
                headers = {
                    "Authorization": f"Bearer {dashscope.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": VISION_MODEL,
                    "input": {"messages": msg},
                    "parameters": {}
                }
                resp = requests.post(
                    "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
                    headers=headers,
                    json=payload,
                    timeout=60  # 60秒超时，避免无限等待
                )
                
                if resp.status_code == 200:
                    res_json = resp.json()
                    if 'output' in res_json and 'choices' in res_json['output']:
                        t = res_json['output']['choices'][0]['message']['content'][0].get('text', '')
                        all_words.extend(re.findall(r'\b[a-zA-Z]{2,}\b', t.lower()))
                        all_chars.extend(re.findall(r'[\u4e00-\u9fa5]', t))
                    else:
                        st.warning(f"第 {i+1} 张图片响应格式异常")
                else:
                    error_msg = resp.json().get('message', resp.text[:200]) if resp.text else f"HTTP {resp.status_code}"
                    st.error(f"第 {i+1} 张识别失败: {error_msg}")
                    has_error = True
                    
            except requests.exceptions.Timeout:
                st.error(f"第 {i+1} 张图片识别超时（60秒），已跳过")
                has_error = True
            except requests.exceptions.ConnectionError:
                st.error(f"第 {i+1} 张图片网络连接失败，请检查网络")
                has_error = True
            except Exception as e:
                st.error(f"第 {i+1} 张识别出错: {type(e).__name__}: {str(e)}")
                has_error = True
        
        progress_bar.progress(1.0, text="识别完成！")
        
        if all_words or all_chars:
            st.session_state.ocr_words = ", ".join(sorted(set(all_words)))
            st.session_state.ocr_chars = " ".join(sorted(set(all_chars)))
            st.session_state.ocr_done = True
            st.rerun()
        elif not has_error:
            st.warning("未识别到任何文字，请检查图片内容")
        else:
            st.error("识别过程中出现错误，请重试")
    
    # 第二步：编辑确认
    if st.session_state.ocr_done:
        st.info("识别完成！请检查并修改结果，确认后保存。")
        with st.form("ocr_confirm_form"):
            edited_words = st.text_area("识别到的单词（可编辑）", value=st.session_state.ocr_words, height=80)
            edited_chars = st.text_area("识别到的汉字（可编辑）", value=st.session_state.ocr_chars, height=80)
            
            col_save, col_cancel = st.columns(2)
            submitted = col_save.form_submit_button("✅ 确认保存", use_container_width=True)
            cancelled = col_cancel.form_submit_button("❌ 放弃", use_container_width=True)
            
            if submitted:
                w_list = [w.strip() for w in re.split(r'[,\s\n]+', edited_words) if w.strip()]
                c_list = [c.strip() for c in re.split(r'[,\s\n]+', edited_chars) if c.strip()]
                if w_list or c_list:
                    append_history(w_list, c_list)
                    st.success(f"已保存: {len(w_list)} 个单词, {len(c_list)} 个汉字")
                else:
                    st.warning("没有内容需要保存")
                st.session_state.ocr_done = False
                st.session_state.ocr_words = ""
                st.session_state.ocr_chars = ""
                st.rerun()
            
            if cancelled:
                st.session_state.ocr_done = False
                st.session_state.ocr_words = ""
                st.session_state.ocr_chars = ""
                st.rerun()

st.divider()

st.subheader("🧚 施展魔法 · 生成绘本")
ca, cb = st.columns(2)

def handle_gen(items, lang):
    with st.status("✨ 正在施法...", expanded=True) as s:
        s.write("📚 正在写故事...")
        story = generate_story(items, lang)
        if story:
            st.session_state.story_content = story
            s.write("🎨 正在画画 (约30秒)...")
            imgs = generate_images(story)
            st.session_state.story_imgs = imgs
            s.update(label="✅ 绘本完成！", state="complete")
        else: st.error("故事生成失败")

if ca.button("生成中文绘本 🏮", use_container_width=True):
    if len(chars_list) < 3: st.warning("汉字不够哦")
    else: handle_gen(chars_list, 'zh')

if cb.button("Generate English Book 📖", use_container_width=True):
    if len(words_list) < 3: st.warning("Words needed")
    else: handle_gen(words_list, 'en')

if st.session_state.story_content:
    st.markdown(f'<div class="story-box">{st.session_state.story_content}</div>', unsafe_allow_html=True)
    if st.session_state.story_imgs:
        cols = st.columns(len(st.session_state.story_imgs))
        for idx, bdata in enumerate(st.session_state.story_imgs):
            cols[idx].image(f"data:image/png;base64,{bdata}", use_container_width=True, caption=f"插画 {idx+1}")
    
    if st.button("关闭绘本"):
        st.session_state.story_content = ""; st.session_state.story_imgs = []; st.rerun()

st.divider()

# 局域网访问二维码
with st.expander("📱 手机扫码访问"):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        url = f"http://{local_ip}:8501"
        st.write(f"**局域网地址:** `{url}`")
        img = qrcode.make(url)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        st.image(buf.getvalue(), width=200, caption="手机扫码访问")
    except Exception as e:
        st.warning(f"无法获取局域网 IP: {e}")

with st.expander("🛠️ API 诊断日志"):
    if 'debug_api_log' in st.session_state:
        for log in st.session_state.debug_api_log: st.code(log)
