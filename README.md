# LLM-

陪伴机器人训练与 Streamlit 使用（emention_bot）

本目录用于：**训练/微调自己的 LoRA 模型**，并在 **Streamlit** 中加载对话使用。

---

## 一、目录下应有哪些文件

建议 **emention_bot** 下保持如下结构，所有路径已改为基于本目录的相对路径，本地和远程都可直接运行：

```
emention_bot/
├── README.md              # 本说明（可选）
├── train.py               # 训练脚本（Unsloth + LoRA）
├── train.jsonl            # 训练数据（messages 格式：system/user/assistant）
├── streamlit_app.py       # Streamlit 主应用（对话界面）
├── start_streamlit.py     # 启动 Streamlit 的入口脚本
├── export_gguf.py         # 可选：合并 LoRA 导出 GGUF 全量模型
├── main.py                # 可选：占位脚本，与训练/Streamlit 无关
├── lora_model/            # LoRA 权重目录（训练结束后保存到这里）
│   ├── adapter_model.safetensors
│   ├── adapter_config.json
│   ├── tokenizer.json
│   └── ...
└── outputs/               # 训练过程产生的 checkpoint（可选，与 lora_model 二选一使用）
    └── checkpoint-60/
        ├── adapter_model.safetensors
        └── ...
```

---

## 二、需要“放到 emention_bot 下面”的内容

| 内容 | 说明 |
|------|------|
| **训练数据** | `train.jsonl` 已在 emention_bot 下；格式为每行一条 `{"messages": [{"role":"system/user/assistant", "content":"..."}]}`。 |
| **LoRA 权重** | 训练完成后会自动保存到 **emention_bot/lora_model/**。若之前在项目根目录的 `outputs/checkpoint-60` 训练过，可把该目录**复制到** emention_bot 下：`emention_bot/outputs/checkpoint-60`，Streamlit 会优先从这里加载。 |
| **Streamlit 相关** | `streamlit_app.py`、`start_streamlit.py` 已在 emention_bot 下，无需再移动。 |

**若希望“全部在 emention_bot 下”**：把项目根目录的 **outputs/** 整个复制到 emention_bot 下即可：

```bash
# 在项目根目录执行
xcopy /E /I outputs emention_bot\outputs
```

之后训练也会把 checkpoint 写到 **emention_bot/outputs/**。

---

## 三、各文件功能简述

| 文件 | 功能 |
|------|------|
| **train.py** | 用 `train.jsonl` 做 SFT，保存 LoRA 到 `lora_model/`，checkpoint 到 `outputs/`。数据格式已支持 messages（system/user/assistant）。 |
| **streamlit_app.py** | Streamlit 对话页：可选「LoRA 微调模型」或「基础模型」，会按顺序查找 `lora_model`、`outputs/checkpoint-60` 等路径。 |
| **start_streamlit.py** | 以脚本所在目录为工作目录，安装依赖并执行 `streamlit run streamlit_app.py`，本地/远程通用。 |
| **export_gguf.py** | 将 base 模型 + LoRA 合并并保存为全量模型；路径目前为远程 `/root/autodl-tmp/...`，本地使用需自行改路径。 |

---

## 四、推荐使用方式

1. **训练**（在 emention_bot 目录下或从项目根指定该目录）：
   ```bash
   cd emention_bot
   python train.py
   ```
2. **启动 Streamlit**：
   ```bash
   cd emention_bot
   python start_streamlit.py
   ```
   或：
   ```bash
   cd emention_bot
   streamlit run streamlit_app.py
   ```

这样所有逻辑和数据都在 **emention_bot** 下，结构清晰，便于单独拷贝或部署。

---

## 五、让 GGUF 使用 5090 显卡（解决 CPU 占满）

若在 AutoDL 上选「本地微调模型」加载 GGUF 时 **CPU 占用很高、5090 几乎不用**，说明当前安装的是 **CPU 版** `llama-cpp-python`，需要换成 **带 CUDA 的版本**。

**步骤（在 SSH 终端执行）：**

1. **卸载当前版本**
   ```bash
   pip uninstall -y llama-cpp-python
   ```

2. **安装 CUDA 12 预编译版（推荐，约 550MB，需耐心下载）**  
   安装前先开启学术加速并设置 pip 国内源，再执行安装：
   ```bash
   # 1. 开启学术加速（AutoDL 等环境可用）
   source /etc/network_turbo

   # 2. 设置 pip 使用清华源（加速其他依赖下载）
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

   # 3. 安装 CUDA 版（wheel 来自 GitHub，学术加速有助于下载）
   pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
   ```
   **若下载到一半报 `Read timed out`**：加大超时后重试，例如：
   ```bash
   pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 --timeout 600
   ```
   （`--timeout 600` 表示 600 秒内未收到数据才判超时，可按网络情况再调大。）

3. **若上面源太慢，可尝试从源码编译（需环境里有 `nvcc`）**
   ```bash
   export PATH=/usr/local/cuda/bin:$PATH
   CMAKE_ARGS="-DGGML_CUDA=ON -DGGML_OPENMP=OFF" pip install llama-cpp-python --no-cache-dir
   ```

4. **安装完成后重启 Streamlit**（建议用 `python start_streamlit.py`，脚本已自动优先使用系统 libstdc++），再选「本地微调模型」加载，此时应看到 GPU 占用、推理明显变快。

**若加载 GGUF 报错 `GLIBCXX_3.4.30' not found`**：说明 Conda 自带的 libstdc++ 过旧。用 `python start_streamlit.py` 启动即可（脚本会优先使用系统 `/usr/lib/x86_64-linux-gnu` 下的 libstdc++）；若直接运行 `streamlit run ...`，可先执行：
   ```bash
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   ```

---

### 若直接 `pip install llama-cpp-python` 报错（OpenMP 链接失败）

在 **Conda 环境**里从源码构建 CPU 版时，可能出现：

- `libgomp.so.1, needed by bin/libggml-cpu.so, not found`
- `undefined reference to 'GOMP_barrier'` / `omp_get_thread_num` 等

**原因**：Conda 的 `compiler_compat` 链接器找不到系统的 OpenMP 库。

**建议**：优先用上面的 **CUDA 预编译 wheel**（步骤 2），不依赖本地编译。若只需 CPU 版且必须从源码装，可先改用系统编译器并确保链接 libgomp，例如：

```bash
# 使用系统 gcc/g++，并显式链接 OpenMP（需已安装 libgomp，一般系统自带）
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu -lgomp"
pip install llama-cpp-python --no-cache-dir
```

若仍失败，可关闭 OpenMP 再编译（CPU 性能会略降）：  
`CMAKE_ARGS="-DGGML_OPENMP=OFF" pip install llama-cpp-python --no-cache-dir`
