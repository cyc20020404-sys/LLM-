import os

# 必须在任何会触发 HuggingFace 下载的 import 之前设置，否则 RAG 嵌入模型 BAAI/bge-small-zh-v1.5 会直连 huggingface.co 报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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

# 若 HuggingFace 无法访问或下载超时，则强制让 Unsloth 走 ModelScope
# 需要在当前环境安装：pip install modelscope
os.environ["UNSLOTH_USE_MODELSCOPE"] = "1"

# 保证本应用目录在 path 最前，使「rag_modules」解析为 emention_bot/rag_modules（含 load_documents 等），避免被 graph_rag_deepseek/rag_modules 抢占
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from rag_modules import (
    load_documents,
    chunk_documents,
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    need_rebuild,
    save_fingerprint,
    HybridRetriever,
    format_context,
)

def _log(msg):
    """输出到终端，便于在无浏览器日志时排查"""
    print(f"[streamlit 模型] {msg}", flush=True)

# Unsloth 仅在加载 HF 模型时按需导入（需 GPU）；使用 GGUF 时不导入，避免无 GPU 环境报错

# 本地微调模型：优先 GGUF 单文件，其次为 HF 格式目录（_APP_DIR 已在上方定义）
# RAG 知识库与索引目录
KNOWLEDGE_DIR = os.path.join(_APP_DIR, "knowledge_base")
RAG_INDEX_DIR = os.path.join(_APP_DIR, "rag_index")
# GGUF 单文件（可配置多个 .gguf 路径，按顺序尝试）
LOCAL_GGUF_FILES = [
    os.path.join(_APP_DIR, "my_emotional_bot.Q4_K_M.gguf"),
]
# HF 格式目录（合并后的 safetensors 等）
LOCAL_MODEL_DIRS = [
    os.path.join(_APP_DIR, "merged_model"),
]
# DPO 微调模型：
# - 优先：merged_model_dpo（已合并权重，完全离线加载）
# - 备选：lora_model_dpo（LoRA 适配器，需要本地已有基座模型或可联网）
LOCAL_DPO_MERGED_DIR = os.path.join(_APP_DIR, "merged_model_dpo")
LOCAL_DPO_LORA_DIR = os.path.join(_APP_DIR, "lora_model_dpo")
# 图 RAG 烹饪助手（Neo4j + Milvus + DeepSeek），独立子项目
GRAPH_RAG_DIR = os.path.join(_APP_DIR, "graph_rag_deepseek")

# 功能模块：左侧一级入口，图RAG 与情感机器人分离
MAIN_MODULE_EMOTION = "情感机器人"
MAIN_MODULE_RAG = "RAG"
MAIN_MODULE_AGENT = "AGENT (敬请期待)"
MAIN_MODULES = [MAIN_MODULE_EMOTION, MAIN_MODULE_RAG, MAIN_MODULE_AGENT]
# 情感机器人下的模型选项（不含图RAG）
EMOTION_MODELS = ["本地微调模型", "DPO微调模型", "基础模型"]

# 语气提示词：需与 DPO 人设（小团团、活泼温柔）兼容，避免指令冲突导致乱讲
TONE_PROMPTS = {
    "无提示词": "",
    "小孩": "【小团团对小朋友】用词简单、句子短，语气温暖耐心，可偶尔用叠词增加趣味，但回答要清晰完整、不要堆砌拟声词。",
    "年轻人": "【当前对话对象：年轻人】请用轻松、自然、像朋友聊天的语气回答：直接不啰嗦，可以带一点日常或网络用语，保持友好和共鸣，像同龄人一样交流。",
    "老年人": "【小团团对长辈】保持尊敬体贴，把话说清楚，多用敬语，少用网络用语和emoji，让对方感到被尊重。语气仍保持温暖，但更稳重。",
}


def _build_chat_prompt(user_input: str, rag_context: str | None = None) -> str:
    """GGUF 等使用：User/Assistant 简单格式，可选注入 RAG 上下文。"""
    tone = st.session_state.get("tone_style", "年轻人")
    instruction = TONE_PROMPTS.get(tone, "")
    ctx_block = ""
    if rag_context:
        ctx_block = (
            "以下是从知识库中检索到的参考资料，请结合这些内容来回答用户的问题：\n"
            "---\n"
            f"{rag_context}\n"
            "---\n\n"
        )
    system_part = ""
    if instruction or ctx_block:
        system_part = f"{instruction}\n\n{ctx_block}" if instruction else ctx_block
    prefix = f"{system_part}User: " if system_part else "User: "
    return f"{prefix}{user_input}\nAssistant:"


# 语气来源说明：
# - 模型（DPO 权重）：从 chosen/rejected 对中学到「活泼温柔 vs 官方生硬」的偏好
# - Prompt（下文人设）：与训练时的 system 一致，提供上下文，使模型知道当前是「小团团」场景
# 二者缺一不可：无 prompt 则模型不知人设，无 DPO 则模型无该偏好。实际语气 = 两者共同作用。
DPO_SYSTEM_PROMPT = (
    "你是小团团，一个活泼温柔、像朋友一样聊天的AI助手。"
    "请用轻松自然的语气回答，可带emoji和网络用语，避免官方、生硬、模板化的表达。"
)

def _build_hf_prompt(user_input: str, tokenizer) -> str:
    """
    HF/DPO 模型专用：使用与 DPO 训练相同的 chat_template 和 system 人设，
    否则模型在推理时看到的格式与训练不一致，无法正确输出活泼语气。
    """
    tone = st.session_state.get("tone_style", "年轻人")
    instruction = TONE_PROMPTS.get(tone, "")
    is_dpo = st.session_state.get("current_model") == "DPO微调模型"

    # 若启用 RAG，先检索知识并构建上下文块
    rag_ctx = ""
    if st.session_state.get("rag_enabled"):
        try:
            retriever = init_rag()
            if retriever is not None:
                top_k = int(st.session_state.get("rag_top_k", 3))
                docs = retriever.search(user_input, top_k=top_k)
                rag_ctx = format_context(docs)
        except Exception as e:
            _log(f"RAG 检索失败: {e}")

    rag_block = ""
    if rag_ctx:
        rag_block = (
            "以下是从知识库中检索到的参考资料，请结合这些内容来回答用户的问题：\n"
            "---\n"
            f"{rag_ctx}\n"
            "---\n\n"
        )

    # DPO 模型：必须注入训练时的人设；有语气选项时与人设合并，再追加 RAG 块
    if is_dpo:
        system_content = DPO_SYSTEM_PROMPT
        if instruction:
            system_content = f"{DPO_SYSTEM_PROMPT}\n\n{instruction}"
        if rag_block:
            system_content = f"{system_content}\n\n{rag_block}"
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_input}]
    elif instruction or rag_block:
        system_parts = []
        if instruction:
            system_parts.append(instruction)
        if rag_block:
            system_parts.append(rag_block)
        system_content = "\n\n".join(system_parts)
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_input}]
    else:
        messages = [{"role": "user", "content": user_input}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt
    except Exception:
        return _build_chat_prompt(user_input, rag_ctx if rag_ctx else None)


@st.cache_resource
def init_rag():
    """初始化 RAG：构建或加载知识库索引，返回 HybridRetriever 或 None。"""
    kb_dir = KNOWLEDGE_DIR
    index_dir = RAG_INDEX_DIR

    if not os.path.isdir(kb_dir):
        _log(f"RAG 知识库目录不存在: {kb_dir}")
        return None

    try:
        # 判断是否需要重建索引
        rebuild = need_rebuild(kb_dir, index_dir)
        from langchain_huggingface import HuggingFaceEmbeddings

        if not rebuild:
            # 加载已有索引
            embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-zh-v1.5",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            vectorstore = load_faiss_index(index_dir, embeddings)
            if vectorstore is not None:
                docs = load_documents(kb_dir)
                chunks = chunk_documents(docs)
                _log(f"RAG 索引已加载，chunk 数量: {len(chunks)}")
                return HybridRetriever(vectorstore, chunks)
            else:
                rebuild = True

        # 需要重建索引
        docs = load_documents(kb_dir)
        if not docs:
            _log("RAG 知识库为空或加载失败")
            return None
        chunks = chunk_documents(docs)
        vectorstore, embeddings = build_faiss_index(
            chunks, embedding_model="BAAI/bge-small-zh-v1.5"
        )
        save_faiss_index(vectorstore, index_dir)
        save_fingerprint(kb_dir, index_dir)
        _log(f"RAG 索引重建完成，chunk 数量: {len(chunks)}")
        return HybridRetriever(vectorstore, chunks)
    except Exception as e:
        _log(f"init_rag 异常: {e}")
        return None


def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """快速检测 TCP 端口是否可达（用于判断隧道/服务是否就绪）"""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError, TimeoutError):
        return False


@st.cache_resource
def init_graph_rag():
    """
    初始化图 RAG 烹饪助手（Neo4j + Milvus + DeepSeek）。
    需先在本机或远程可访问的地址启动 Neo4j/Milvus，并配置 graph_rag_deepseek/.env 中的 DEEPSEEK_API_KEY。
    """
    if not os.path.isdir(GRAPH_RAG_DIR):
        _log(f"图 RAG 目录不存在: {GRAPH_RAG_DIR}")
        return None

    # ── 预检：在尝试初始化前先快速探测 Neo4j / Milvus 端口 ──
    from dotenv import load_dotenv
    load_dotenv(os.path.join(GRAPH_RAG_DIR, ".env"))
    if _APP_DIR not in sys.path:
        sys.path.insert(0, _APP_DIR)
    from graph_rag_deepseek.config import DEFAULT_CONFIG

    # 解析 Neo4j host:port（bolt://host:port）
    import re as _re
    _neo4j_m = _re.search(r"://([^:/]+):(\d+)", DEFAULT_CONFIG.neo4j_uri)
    _neo4j_host = _neo4j_m.group(1) if _neo4j_m else "127.0.0.1"
    _neo4j_port = int(_neo4j_m.group(2)) if _neo4j_m else 7687

    _missing = []
    if not _check_port(_neo4j_host, _neo4j_port):
        _missing.append(f"Neo4j ({_neo4j_host}:{_neo4j_port})")
    if not _check_port(DEFAULT_CONFIG.milvus_host, DEFAULT_CONFIG.milvus_port):
        _missing.append(f"Milvus ({DEFAULT_CONFIG.milvus_host}:{DEFAULT_CONFIG.milvus_port})")

    if _missing:
        _log(f"⚠️ 以下服务不可达，请检查 Docker 容器或 SSH 隧道是否正常: {', '.join(_missing)}")
        return None

    # ── 端口可达，正式初始化 ──
    try:
        from graph_rag_deepseek.main import AdvancedGraphRAGSystem
        _log("正在初始化图 RAG 系统（Neo4j + Milvus + DeepSeek）...")
        system = AdvancedGraphRAGSystem(config=DEFAULT_CONFIG)
        system.initialize_system()
        system.build_knowledge_base()
        _log("图 RAG 系统初始化完成")
        return system
    except Exception as e:
        _log(f"init_graph_rag 异常: {e}")
        import traceback
        traceback.print_exc()
        return None


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

            if model_type == "DPO微调模型":
                # 1) 优先：加载已合并权重的 DPO 模型（完全本地，不访问 HuggingFace/ModelScope）
                merged_path = LOCAL_DPO_MERGED_DIR
                if os.path.isdir(merged_path):
                    _log(f"尝试加载 DPO 合并模型: {merged_path}")
                    try:
                        from unsloth import FastLanguageModel
                        model, tokenizer = FastLanguageModel.from_pretrained(
                            merged_path,
                            max_seq_length=max_seq_length,
                            load_in_4bit=False,  # 已是 16bit 合并权重，直接加载
                        )
                        model = FastLanguageModel.for_inference(model)
                        _log(f"DPO 合并模型加载成功: {merged_path}")
                        st.success(f"✅ DPO 合并模型加载成功：{merged_path}")
                        return model, tokenizer, "hf"
                    except Exception as e:
                        _log(f"DPO 合并模型加载失败: {e}")
                        st.warning(f"⚠️ DPO 合并模型加载失败，将尝试 LoRA 目录：{str(e)}")

                # 2) 退而求其次：加载 LoRA 适配器目录（可能仍需访问基座模型）
                lora_path = LOCAL_DPO_LORA_DIR
                if not os.path.isdir(lora_path):
                    _log(f"DPO LoRA 目录不存在: {lora_path}")
                    st.error(
                        "❌ 未找到 DPO 模型目录：\n"
                        f"- 合并目录：{merged_path}\n"
                        f"- LoRA 目录：{lora_path}\n\n"
                        "请先在项目根目录运行 `python train_dpo.py` 完成 DPO 训练。"
                    )
                    return None, None, None
                _log(f"尝试加载 DPO LoRA 模型目录: {lora_path}")
                try:
                    from unsloth import FastLanguageModel
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        lora_path,
                        max_seq_length=max_seq_length,
                        load_in_4bit=True,  # 与训练时基座一致，节省显存
                    )
                    model = FastLanguageModel.for_inference(model)
                    _log(f"DPO LoRA 模型加载成功: {lora_path}")
                    st.success(f"✅ DPO LoRA 微调模型加载成功：{lora_path}")
                    return model, tokenizer, "hf"
                except Exception as e:
                    _log(f"DPO LoRA 模型加载失败: {e}")
                    st.error(f"❌ DPO 模型加载失败: {str(e)}")
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

# 尽早初始化 load_status 与 session state，避免侧栏/主区读取时报 KeyError
if "load_status" not in st.session_state:
    st.session_state.load_status = "未加载（发送消息或点击「预加载模型」后加载）"
if "main_module" not in st.session_state:
    st.session_state.main_module = MAIN_MODULE_EMOTION
if "messages_emotion" not in st.session_state:
    st.session_state.messages_emotion = getattr(st.session_state, "messages", []) if "messages" in st.session_state else []
if "messages_rag" not in st.session_state:
    st.session_state.messages_rag = []
if "messages" in st.session_state and not st.session_state.messages_emotion and st.session_state.messages:
    st.session_state.messages_emotion = list(st.session_state.messages)
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "backend" not in st.session_state:
    st.session_state.backend = "hf"
if "graph_rag_system" not in st.session_state:
    st.session_state.graph_rag_system = None

# 主区标题与状态条（按模块切换）
_main_module = st.session_state.get("main_module", MAIN_MODULE_EMOTION)
if _main_module == MAIN_MODULE_EMOTION:
    st.title("🤖 情感机器人")
elif _main_module == MAIN_MODULE_RAG:
    st.title("🕸️ 图RAG 烹饪助手")
else:
    st.title("🤖 Agent")
# 状态条：左侧状态，右侧操作
_status_col1, _status_col2 = st.columns([3, 1])
with _status_col1:
    st.caption("📌 " + st.session_state.load_status)
with _status_col2:
    pass  # 按钮放到下面和清空/回到底部一起
st.markdown("---")

# 侧边栏：功能模块一级入口 + 方案 A 分组
with st.sidebar:
    st.header("⚙️ 功能")
    main_module = st.radio(
        "选择功能模块",
        options=MAIN_MODULES,
        index=MAIN_MODULES.index(st.session_state.main_module) if st.session_state.main_module in MAIN_MODULES else 0,
        key="main_module",
        label_visibility="collapsed",
    )

    # ---------- 情感机器人：方案 A 分组 ----------
    if main_module == MAIN_MODULE_EMOTION:
        st.subheader("🤖 情感机器人")
        # 第一块（常开）：模型选择、状态、预加载
        model_choice = st.selectbox(
            "选择模型",
            EMOTION_MODELS,
            key="emotion_model_choice",
            help="本地微调 / DPO / 基础模型。",
        )
        st.caption(f"📌 **{st.session_state.load_status}**")
        if model_choice == "基础模型":
            st.caption("⚠️ 需 PyTorch 能见 GPU，否则会加载失败")
        elif model_choice == "DPO微调模型":
            _dpo_exists = os.path.isdir(LOCAL_DPO_MERGED_DIR) or os.path.isdir(LOCAL_DPO_LORA_DIR)
            st.caption(
                f"{'✅' if _dpo_exists else '❌'} DPO 目录 "
                f"{'已就绪' if _dpo_exists else '未找到，请先运行 train_dpo.py'}"
            )
        if st.button("🔄 预加载模型", use_container_width=True, type="primary", key="btn_preload_emotion"):
            _log(f"预加载情感模型: {model_choice}")
            model, tokenizer, backend = load_model(model_choice)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.backend = backend or "hf"
            st.session_state.graph_rag_system = None
            st.session_state.model_loaded = True
            st.session_state.current_model = model_choice
            st.session_state.load_status = f"✅ 已加载（{model_choice}）" if model else "❌ 加载失败，请查看终端日志"
            st.rerun()

        with st.expander("📝 对话设置", expanded=False):
            tone_style = st.selectbox(
                "语气风格（对话对象）",
                options=["无提示词", "小孩", "年轻人", "老年人"],
                index=0,
                key="tone_style",
                help="无提示词 = 不加语气指令；其余会注入对应提示词。",
            )
            rag_enabled = st.checkbox(
                "启用 RAG 知识检索",
                value=False,
                help="从 knowledge_base/ 检索资料作为回答依据。",
                key="rag_enabled",
            )
            rag_top_k = st.slider(
                "RAG 检索 Top-K",
                min_value=1,
                max_value=5,
                value=3,
                step=1,
                key="rag_top_k",
            )

        with st.expander("🎚️ 生成参数", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                max_tokens = st.slider(
                    "最大生成长度",
                    min_value=50,
                    max_value=2048,
                    value=500,
                    step=50,
                    key="max_tokens",
                )
            with col2:
                temperature = st.slider(
                    "温度",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.7,
                    step=0.1,
                    key="temperature",
                )
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.05,
                key="top_p",
            )
            st.caption("温度越高越创意，Top P 控制多样性。")

    # ---------- RAG：图RAG 烹饪助手 ----------
    elif main_module == MAIN_MODULE_RAG:
        st.subheader("🕸️ 图RAG 烹饪助手")
        _graph_rag_ready = os.path.isdir(GRAPH_RAG_DIR)
        st.caption(
            f"{'🕸️' if _graph_rag_ready else '❌'} "
            + ("Neo4j+Milvus 需已启动；DeepSeek API 在 graph_rag_deepseek/.env 配置" if _graph_rag_ready else "未找到 graph_rag_deepseek 目录")
        )
        st.caption(f"📌 **{st.session_state.load_status}**")
        if st.button("🔄 预加载图 RAG", use_container_width=True, type="primary", key="btn_preload_rag"):
            with st.spinner("🕸️ 正在初始化图 RAG（Neo4j + Milvus + DeepSeek）..."):
                graph_sys = init_graph_rag()
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.backend = "graph_rag"
            st.session_state.graph_rag_system = graph_sys
            st.session_state.model_loaded = graph_sys is not None
            st.session_state.current_model = MAIN_MODULE_RAG
            st.session_state.load_status = f"✅ 已加载（图RAG）" if graph_sys else "❌ 图 RAG 初始化失败，请检查 Neo4j/Milvus/DEEPSEEK_API_KEY"
            st.rerun()

    # ---------- AGENT：占位 ----------
    else:
        st.subheader("🤖 Agent")
        st.caption("敬请期待")

    # 调试信息（各模块共用，默认收拢）
    with st.expander("🔧 调试信息", expanded=False):
        st.caption(f"📂 项目目录: `{_APP_DIR}`")
        st.caption(f"🕸️ 图RAG 目录: {'✅ 存在' if os.path.isdir(GRAPH_RAG_DIR) else '❌ 不存在'}")
        _cuda_ok = torch.cuda.is_available()
        _cuda_msg = f"是（{torch.cuda.get_device_name(0)}）" if _cuda_ok else "否"
        st.caption(f"PyTorch 可见 GPU: **{_cuda_msg}**")

# 生成参数（情感机器人侧栏中设置，此处取默认供后续逻辑使用）
max_tokens = st.session_state.get("max_tokens", 500)
temperature = st.session_state.get("temperature", 0.7)
top_p = st.session_state.get("top_p", 0.9)


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
    rag_ctx = ""
    if st.session_state.get("rag_enabled"):
        try:
            retriever = init_rag()
            if retriever is not None:
                top_k = int(st.session_state.get("rag_top_k", 3))
                docs = retriever.search(user_input, top_k=top_k)
                rag_ctx = format_context(docs)
        except Exception as e:
            _log(f"RAG 检索失败(GGUF): {e}")

    prompt = _build_chat_prompt(user_input, rag_ctx if rag_ctx else None)
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
    """HF 模型流式生成，使用 chat_template 与训练格式一致，只显示 </think> 后的回答。"""
    if st.session_state.model is None or st.session_state.tokenizer is None:
        placeholder.markdown("❌ 模型未加载或状态异常")
        return "❌ 模型未加载"
    try:
        from transformers import TextIteratorStreamer
        prompt = _build_hf_prompt(user_input, st.session_state.tokenizer)
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
            repetition_penalty=1.15,  # 抑制呀呀呀、抱抱抱等重复
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
        prompt = (_build_hf_prompt(user_input, st.session_state.tokenizer)
                  if st.session_state.tokenizer else _build_chat_prompt(user_input))

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

        # HF 后端：只解码新生成部分，并去除 <think> 内容
        if st.session_state.tokenizer is None:
            return "❌ 模型状态异常"
        enc = st.session_state.tokenizer(prompt, return_tensors="pt")
        input_len = enc["input_ids"].shape[1]
        enc = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in enc.items()}
        with torch.no_grad():
            outputs = st.session_state.model.generate(
                **enc,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=1.15,
                pad_token_id=st.session_state.tokenizer.pad_token_id,
                eos_token_id=st.session_state.tokenizer.eos_token_id,
            )
        response = st.session_state.tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )
        response = _strip_think_tags(response)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        return response.strip() or "（无输出）"
    except Exception as e:
        return f"❌ 生成失败: {str(e)}"


# 主界面：按模块显示对话区或占位
_main_module = st.session_state.get("main_module", MAIN_MODULE_EMOTION)
_current_messages = (
    st.session_state.messages_emotion
    if _main_module == MAIN_MODULE_EMOTION
    else (st.session_state.messages_rag if _main_module == MAIN_MODULE_RAG else [])
)

col1, col2 = st.columns([3, 1])
with col1:
    if _main_module != MAIN_MODULE_AGENT:
        st.caption("💬 对话")
with col2:
    if st.button("🔄 清空对话", use_container_width=True, key="btn_clear_chat"):
        if _main_module == MAIN_MODULE_EMOTION:
            st.session_state.messages_emotion = []
        elif _main_module == MAIN_MODULE_RAG:
            st.session_state.messages_rag = []
        st.rerun()

# AGENT 占位
if _main_module == MAIN_MODULE_AGENT:
    st.info("🚧 Agent 功能敬请期待。")
else:
    # 显示当前模块对话历史
    for message in _current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 「回到底部」与输入框（仅情感/RAG 显示）
if _main_module != MAIN_MODULE_AGENT:
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
        if st.button("⬇️ 回到底部", use_container_width=True, help="滚动到最新一条消息", key="btn_scroll_bottom"):
            st.session_state.do_auto_scroll = True
            st.rerun()

user_input = st.chat_input("请输入您的问题或想说的话..." if _main_module != MAIN_MODULE_AGENT else "当前模块暂无输入")

if user_input:
    _main_module = st.session_state.get("main_module", MAIN_MODULE_EMOTION)

    if _main_module == MAIN_MODULE_RAG:
        # 图 RAG 烹饪助手
        if not st.session_state.model_loaded or st.session_state.current_model != MAIN_MODULE_RAG:
            with st.spinner("🕸️ 正在初始化图 RAG..."):
                graph_sys = init_graph_rag()
            st.session_state.graph_rag_system = graph_sys
            st.session_state.model_loaded = graph_sys is not None
            st.session_state.current_model = MAIN_MODULE_RAG
            st.session_state.load_status = f"✅ 已加载（图RAG）" if graph_sys else "❌ 图 RAG 初始化失败，请查看终端"
        st.session_state.messages_rag.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        graph_sys = st.session_state.get("graph_rag_system")
        if graph_sys is not None:
            with st.chat_message("assistant"):
                with st.spinner("🕸️ 图 RAG 检索与生成中..."):
                    result, analysis = graph_sys.ask_question_with_routing(user_input, stream=False)
                st.markdown(result or "（无回复）")
                if analysis is not None:
                    with st.expander("📊 本次检索策略"):
                        st.caption(
                            f"**策略**: {analysis.recommended_strategy.value} | "
                            f"复杂度: {analysis.query_complexity:.2f} | 关系密集度: {analysis.relationship_intensity:.2f}"
                        )
            st.session_state.messages_rag.append({"role": "assistant", "content": result or "（无回复）"})
        else:
            st.error("❌ 图 RAG 未就绪，请先点击「预加载图 RAG」或检查 Neo4j/Milvus/DEEPSEEK_API_KEY")
        st.rerun()

    elif _main_module == MAIN_MODULE_EMOTION:
        # 情感机器人：本地/DPO/基础模型
        model_choice = st.session_state.get("emotion_model_choice", EMOTION_MODELS[0])
        if not st.session_state.model_loaded or st.session_state.current_model != model_choice:
            model, tokenizer, backend = load_model(model_choice)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.backend = backend or "hf"
            st.session_state.graph_rag_system = None
            st.session_state.model_loaded = True
            st.session_state.current_model = model_choice
            st.session_state.load_status = f"✅ 已加载（{model_choice}）" if model else "❌ 加载失败，请查看终端"
        st.session_state.messages_emotion.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        if st.session_state.model is not None:
            backend = st.session_state.get("backend", "hf")
            if backend == "gguf":
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response = stream_gguf_response(
                        user_input, placeholder,
                        max_tok=max_tokens,
                        temp=temperature,
                        top_p_val=top_p,
                    )
            else:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    response = stream_hf_response(
                        user_input, placeholder,
                        max_tok=max_tokens,
                        temp=temperature,
                        top_p_val=top_p,
                    )
            st.session_state.messages_emotion.append({"role": "assistant", "content": response})
        else:
            st.error("❌ 模型加载失败，无法生成响应")
        st.rerun()
    else:
        # AGENT 占位：忽略输入或可提示
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
