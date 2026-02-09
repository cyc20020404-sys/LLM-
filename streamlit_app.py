import os
import re
import sys
from threading import Thread

# 优先使用系统 libstdc++，避免 Conda 下 llama-cpp-python 报 GLIBCXX_3.4.30 not found（须在 import llama_cpp 之前）
for _path in ("/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
    if os.path.isdir(_path):
        _prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = _path + (":" + _prev if _prev else "")
        break

import streamlit as st
import torch
import time

def _log(msg):
    """输出到终端，便于在无浏览器日志时排查"""
    print(f"[streamlit 模型] {msg}", flush=True)

# Unsloth 仅在加载 HF 模型时按需导入（需 GPU）；使用 GGUF 时不导入，避免无 GPU 环境报错

# 配置环境
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# 本地微调模型：优先 GGUF 单文件，其次为 HF 格式目录
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
# GGUF 单文件（可配置多个 .gguf 路径，按顺序尝试）
LOCAL_GGUF_FILES = [
    os.path.join(_APP_DIR, "my_emotional_bot.Q4_K_M.gguf"),
]
# HF 格式目录（合并后的 safetensors 等）
LOCAL_MODEL_DIRS = [
    os.path.join(_APP_DIR, "merged_model"),
]

# 三套语气提示词：小孩 / 年轻人 / 老年人
TONE_PROMPTS = {
    "小孩": "【当前对话对象：小朋友】请用对小孩说话的方式回答：用词简单、句子短，语气温暖耐心、带一点鼓励和趣味，避免复杂概念，可以适当用叠词或拟声词，让对话更亲切可爱。",
    "年轻人": "【当前对话对象：年轻人】请用轻松、自然、像朋友聊天的语气回答：直接不啰嗦，可以带一点日常或网络用语，保持友好和共鸣，像同龄人一样交流。",
    "老年人": "【当前对话对象：长辈/老年人】请用尊敬、耐心、体贴的方式回答：把话说清楚、语速放慢的感觉，多用敬语和关心，避免网络用语和复杂概念，让对方感到被尊重和照顾。",
}


def _build_chat_prompt(user_input: str) -> str:
    """根据当前选择的语气风格，拼接带语气指令的对话 prompt。"""
    tone = st.session_state.get("tone_style", "年轻人")
    instruction = TONE_PROMPTS.get(tone, TONE_PROMPTS["年轻人"])
    return f"{instruction}\n\nUser: {user_input}\nAssistant:"


# 加载模型（须在侧边栏“预加载模型”按钮之前定义）
@st.cache_resource
def load_model(model_type):
    """加载模型。本地微调：优先 GGUF 单文件，否则 HF 目录；不加载基础模型。"""
    max_seq_length = 4096
    _log(f"开始加载模型，类型: {model_type}")
    with st.spinner("🔄 正在加载模型，请稍候..."):
        try:
            if model_type == "本地微调模型":
                # 1) 优先：GGUF 单文件
                for path in LOCAL_GGUF_FILES:
                    if not os.path.isfile(path):
                        _log(f"GGUF 不存在，跳过: {path}")
                        continue
                    _log(f"尝试加载 GGUF: {path}")
                    try:
                        from llama_cpp import Llama
                        llm = Llama(
                            model_path=path,
                            n_ctx=max_seq_length,
                            n_gpu_layers=-1,
                            verbose=False,
                        )
                        _log(f"GGUF 加载成功: {path}")
                        # 检测是否为 CUDA 版（当前若为 CPU 版会导致 GPU 0%、CPU 100% 很慢）
                        try:
                            from llama_cpp.llama_cpp import _load_shared_library
                            _lib = _load_shared_library("llama")
                            if getattr(_lib, "llama_supports_gpu_offload", lambda: False)():
                                _log("当前 llama-cpp-python 支持 GPU 卸载，推理应走 GPU")
                            else:
                                _log("当前为 CPU 版 llama-cpp-python，推理会非常慢；请安装 CUDA 版（见 README）")
                                st.warning("⚠️ 当前为 **CPU 版** llama-cpp-python，推理时 GPU 会显示 0%、CPU 满负载很慢。请安装 CUDA 版后重启应用，见 README「让 GGUF 使用 5090 显卡」。")
                        except Exception:
                            pass
                        st.success(f"✅ 本地 GGUF 模型加载成功：{path}")
                        return llm, None, "gguf"
                    except ImportError as e:
                        _log(f"ImportError: {e}")
                        st.error("❌ 请先安装 llama-cpp-python：pip install llama-cpp-python（GPU 版需带 CUDA 编译）")
                        return None, None, None
                    except Exception as e:
                        err = str(e)
                        _log(f"GGUF 加载异常 {path}: {err}")
                        if "not within the file bounds" in err or "corrupted or incomplete" in err:
                            st.warning(f"⚠️ GGUF 文件已损坏或不完整，请重新导出或下载：{path}")
                        else:
                            st.warning(f"⚠️ GGUF {path} 加载失败: {err}")
                        continue
                # 2) 备选：HF 格式目录（需 GPU + Unsloth）
                for path in LOCAL_MODEL_DIRS:
                    if not os.path.exists(path):
                        _log(f"HF 目录不存在，跳过: {path}")
                        continue
                    _log(f"尝试加载 HF 目录: {path}")
                    try:
                        from unsloth import FastLanguageModel
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            path,
                            max_seq_length=max_seq_length,
                            load_in_4bit=False,
                        )
                        model = FastLanguageModel.for_inference(model)
                        _log(f"HF 加载成功: {path}")
                        st.success(f"✅ 本地微调模型加载成功：{path}")
                        return model, tokenizer, "hf"
                    except Exception as e:
                        _log(f"HF 加载失败 {path}: {e}")
                        st.warning(f"⚠️ {path} 加载失败: {str(e)}")
                        continue
                _log("未找到可用的本地模型")
                st.error("❌ 未找到可用的本地模型（GGUF 文件或 merged_model 等目录）。")
                return None, None, None

            # 基础模型：从 HuggingFace 加载（需 GPU + Unsloth）
            _log("加载基础模型（HuggingFace）...")
            try:
                from unsloth import FastLanguageModel
                model_name = "unsloth/deepseek-r1-distill-qwen-7b-unsloth-bnb-4bit"
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=max_seq_length,
                    load_in_4bit=True,
                )
                model = FastLanguageModel.for_inference(model)
                _log("基础模型加载成功")
                st.success("✅ 基础模型加载成功！")
                return model, tokenizer, "hf"
            except Exception as e:
                err = str(e)
                if "torch accelerator" in err.lower() or "need a gpu" in err.lower() or "cuda" in err.lower():
                    _log(f"基础模型加载失败（PyTorch 未见 GPU）: {e}")
                    cuda_visible = torch.cuda.is_available()
                    st.error(
                        "❌ 基础模型依赖 **PyTorch** 的 GPU，当前 PyTorch 未检测到可用 GPU。"
                        "（GGUF 用的是 llama-cpp-python 的 CUDA，和 PyTorch 无关，所以微调模型能跑。）\n\n"
                        "**解决办法**：安装 PyTorch 的 CUDA 版后再试。AutoDL 常见为 CUDA 12.8：\n"
                        "`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`"
                    )
                    return None, None, None
                raise

        except Exception as e:
            _log(f"模型加载异常: {e}")
            st.error(f"❌ 模型加载失败: {str(e)}")
            return None, None, None


# 设置页面配置
st.set_page_config(
    page_title="AI陪伴机器人",
    page_icon="🤖",
    layout="wide"
)

# 确保主内容区可正常用滚轮上下滚动（不锁死底部）
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { overflow-y: auto !important; }
    .main .block-container { overflow-y: visible !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 尽早初始化 load_status，避免下方 caption 报 AttributeError
if "load_status" not in st.session_state:
    st.session_state.load_status = "未加载（发送消息或点击「预加载模型」后加载）"

st.title("🤖 AI陪伴机器人")
# 模型状态（主区域显示，避免浏览器 AbortSignal 等导致侧栏提示不显示）
st.caption("📌 模型状态：**" + st.session_state.load_status + "**")
st.markdown("---")

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 模型配置")
    
    # 语气风格（对话对象）：小孩 / 年轻人 / 老年人
    tone_style = st.selectbox(
        "语气风格（对话对象）",
        options=["小孩", "年轻人", "老年人"],
        index=1,
        key="tone_style",
        help="选择对话对象后，小团团会用对应语气回答（用词与风格会随之调整）",
    )
    
    # PyTorch 是否可见 GPU（基础模型依赖 PyTorch；GGUF 用 llama-cpp-python 的 CUDA，二者独立）
    _cuda_ok = torch.cuda.is_available()
    _cuda_msg = f"是（{torch.cuda.get_device_name(0)}）" if _cuda_ok else "否"
    st.caption(f"🔧 PyTorch 可见 GPU: **{_cuda_msg}**")
    if not _cuda_ok:
        st.caption("系统有 CUDA 但 PyTorch 可能是 CPU 版，需重装 cu128 版；GGUF 不受影响")
    
    # 模型选择
    model_choice = st.selectbox(
        "选择模型加载方式",
        ["本地微调模型", "基础模型"],
        help="基础模型需 GPU；无 GPU 时请选「本地微调模型」并加载 GGUF。",
    )
    if model_choice == "基础模型":
        st.caption("⚠️ 需 PyTorch 能见 GPU，否则会加载失败")
    
    # 生成参数配置
    col1, col2 = st.columns(2)
    with col1:
        max_tokens = st.slider(
            "最大生成长度",
            min_value=50,
            max_value=2048,
            value=500,
            step=50
        )
    
    with col2:
        temperature = st.slider(
            "温度(Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1
        )
    
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.05
    )
    
    st.markdown("---")
    # 预加载按钮：先加载模型并看终端日志，避免“发消息才加载”且前端不显示的问题
    if st.button("🔄 预加载模型", use_container_width=True, type="primary"):
        _log(f"用户点击预加载，当前选择: {model_choice}")
        model, tokenizer, backend = load_model(model_choice)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.backend = backend or "hf"
        st.session_state.model_loaded = True
        st.session_state.current_model = model_choice
        if model is not None:
            st.session_state.load_status = f"✅ 已加载（{model_choice}）"
            _log("预加载成功")
        else:
            st.session_state.load_status = "❌ 加载失败，请查看终端日志"
            _log("预加载失败")
    st.markdown("---")
    st.markdown("**💡 参数说明:**")
    st.markdown("""
    - **温度**: 值越高越具有创意，越低越保守
    - **Top P**: 控制生成多样性的参数
    """)


# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.current_model = None
    st.session_state.backend = "hf"  # "hf" | "gguf"


def _strip_think_tags(text: str) -> str:
    """只保留 </think> 之后的回答部分，去掉 <think>...</think> 思考内容及末尾自检句（如「请问，这个回应是否符合要求？」）。"""
    if not text:
        return ""
    # 按第一个 </think> 截断，只保留后面的正式回答
    if "</think>" in text:
        out = text.split("</think>", 1)[-1].strip()
    else:
        out = re.sub(r"<think>.*", "", text, flags=re.DOTALL).strip()
    # 若回答里又出现 <think>，只保留第一段（到下一个 <think> 或 </think> 之前）
    if "<think>" in out:
        out = re.sub(r"<think>.*", "", out, flags=re.DOTALL).strip()
    # 去掉末尾模型自检句
    out = re.sub(r"请问[，,]?\s*这个回应是否符合要求[？?].*$", "", out, flags=re.DOTALL).strip()
    return out


def stream_gguf_response(user_input, placeholder, max_tok, temp, top_p_val):
    """GGUF 流式生成，边生成边更新 placeholder，返回完整回复。只显示 </think> 后的回答。"""
    prompt = _build_chat_prompt(user_input)
    full = ""
    try:
        stream = st.session_state.model(
            prompt,
            max_tokens=max_tok,
            temperature=temp,
            top_p=top_p_val,
            repeat_penalty=1.15,
            stop=["User:", "\nUser:", "小团团'。。"],
            echo=False,
            stream=True,
        )
        for chunk in stream:
            piece = (chunk.get("choices") or [{}])[0].get("text") or ""
            full += piece
            # 只把 </think> 后的内容展示给用户，避免露出思考过程
            to_show = _strip_think_tags(full)
            placeholder.markdown(to_show + "▌")
        to_show = _strip_think_tags(full)
        placeholder.markdown(to_show)
        if "Assistant:" in full:
            full = full.split("Assistant:")[-1].strip()
        full = _strip_think_tags(full)
        return full.strip() or "（无输出）"
    except Exception as e:
        placeholder.markdown(f"❌ 生成失败: {str(e)}")
        return f"❌ 生成失败: {str(e)}"


def stream_hf_response(user_input, placeholder, max_tok, temp, top_p_val):
    """HF 基础模型流式生成，边生成边更新 placeholder，只显示 </think> 后的回答。"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        placeholder.markdown("❌ 模型未加载或状态异常")
        return "❌ 模型未加载"
    try:
        from transformers import TextIteratorStreamer
        prompt = _build_chat_prompt(user_input)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = st.session_state.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(device)
        streamer = TextIteratorStreamer(
            st.session_state.tokenizer, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp,
            top_p=top_p_val,
            do_sample=True,
            pad_token_id=st.session_state.tokenizer.pad_token_id,
            eos_token_id=st.session_state.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = Thread(target=st.session_state.model.generate, kwargs=gen_kwargs)
        thread.start()
        full = ""
        for new_text in streamer:
            full += new_text
            placeholder.markdown(_strip_think_tags(full) + "▌")
        thread.join()
        full = _strip_think_tags(full)
        placeholder.markdown(full)
        if "Assistant:" in full:
            full = full.split("Assistant:")[-1].strip()
        full = _strip_think_tags(full)
        return full.strip() or "（无输出）"
    except Exception as e:
        placeholder.markdown(f"❌ 生成失败: {str(e)}")
        return f"❌ 生成失败: {str(e)}"


def generate_response(user_input):
    """生成模型响应（支持 HF 与 GGUF 两种后端）"""
    if st.session_state.model is None:
        return "❌ 模型未加载，请先加载模型"
    backend = st.session_state.get("backend", "hf")

    try:
        prompt = _build_chat_prompt(user_input)

        if backend == "gguf":
            # GGUF 使用流式在外部调用，此处仅作非流式兜底（一般不走到）
            with st.spinner("🤔 模型思考中..."):
                out = st.session_state.model(
                    prompt,
                    max_tokens=min(max_tokens, 256),
                    temperature=temperature,
                    top_p=top_p,
                    repeat_penalty=1.15,
                    stop=["User:", "\nUser:", "小团团'。。"],
                    echo=False,
                )
                response = (out["choices"][0].get("text") or "").strip()
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                response = _strip_think_tags(response)
                return response or "（无输出）"

        # HF 后端
        if st.session_state.tokenizer is None:
            return "❌ 模型状态异常"
        inputs = st.session_state.tokenizer.encode(
            prompt, return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=st.session_state.tokenizer.pad_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
            )
        response = st.session_state.tokenizer.decode(
            outputs[0], skip_special_tokens=True
        )
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        return response
    except Exception as e:
        return f"❌ 生成失败: {str(e)}"


# 主界面
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("💬 对话区域")

with col2:
    if st.button("🔄 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 「回到底部」按钮：仅点击时执行一次滚动，不干扰平时用滚轮上下翻看
if st.session_state.get("do_auto_scroll", False):
    try:
        from streamlit.components.v1 import html as st_html
        st_html(
            """
            <script>
            (function() {
                var t = window.parent;
                if (!t || t === window) t = window;
                function run() {
                    try {
                        var app = t.document.querySelector('[data-testid="stAppViewContainer"]');
                        if (app && app.scrollHeight > app.clientHeight) {
                            app.scrollTop = app.scrollHeight;
                        }
                        var main = t.document.querySelector('.main');
                        if (main && main.scrollHeight > main.clientHeight) {
                            main.scrollTop = main.scrollHeight;
                        }
                        t.scrollTo(0, t.document.body.scrollHeight);
                    } catch (e) {}
                }
                setTimeout(run, 100);
            })();
            </script>
            """,
            height=0,
        )
    except Exception:
        pass
    st.session_state.do_auto_scroll = False

with col2:
    if st.button("⬇️ 回到底部", use_container_width=True, help="滚动到最新一条消息"):
        st.session_state.do_auto_scroll = True
        st.rerun()

# 输入框
user_input = st.chat_input("请输入您的问题或想说的话...")

if user_input:
    # 加载模型（如果还未加载）
    if not st.session_state.model_loaded or st.session_state.current_model != model_choice:
        model, tokenizer, backend = load_model(model_choice)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.session_state.backend = backend or "hf"
        st.session_state.model_loaded = True
        st.session_state.current_model = model_choice
        st.session_state.load_status = f"✅ 已加载（{model_choice}）" if model else "❌ 加载失败，请查看终端"
    
    # 显示用户消息
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 生成AI响应
    if st.session_state.model is not None:
        backend = st.session_state.get("backend", "hf")
        if backend == "gguf":
            with st.chat_message("assistant"):
                placeholder = st.empty()
                response = stream_gguf_response(
                    user_input, placeholder,
                    max_tok=min(max_tokens, 512),
                    temp=temperature,
                    top_p_val=top_p,
                )
        else:
            # HF 基础模型也流式输出
            with st.chat_message("assistant"):
                placeholder = st.empty()
                response = stream_hf_response(
                    user_input, placeholder,
                    max_tok=min(max_tokens, 512),
                    temp=temperature,
                    top_p_val=top_p,
                )
        
        # 保存到对话历史
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })
    else:
        st.error("❌ 模型加载失败，无法生成响应")
    
    st.rerun()

# 底部信息
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>🚀 驱动模型: DeepSeek R1 Distill Qwen 7B</small>
    <br>
    <small>⚡ 框架: Unsloth + Streamlit</small>
</div>
""", unsafe_allow_html=True)
