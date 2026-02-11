#!/usr/bin/env python3
"""
直接在远程运行的Streamlit应用启动脚本
将此脚本上传到远程服务器 /root/autodl-tmp/start_streamlit.py
然后运行: python start_streamlit.py
"""

import os
import subprocess
import sys

def main():
    # 优先使用系统 libstdc++，避免 Conda 下 llama-cpp-python 报 GLIBCXX_3.4.30 not found
    system_lib = "/usr/lib/x86_64-linux-gnu"
    if os.path.isdir(system_lib):
        prev = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = system_lib + (":" + prev if prev else "")

    print("=" * 60)
    print("🚀 Streamlit 应用启动脚本")
    print("=" * 60)
    print()

    # 工作目录：优先使用脚本所在目录（emention_bot），便于本地和远程统一
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    print(f"📂 工作目录: {os.getcwd()}")
    print()
    
    # 1. 安装依赖（基础 + RAG + ModelScope）
    print("📦 安装依赖...")
    print("-" * 60)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                # 基础依赖
                "streamlit",
                "torch",
                "transformers",
                "datasets",
                "peft",
                # RAG 依赖
                "langchain",
                "langchain-huggingface",
                "langchain-community",
                "langchain-text-splitters",
                "faiss-cpu",
                "rank-bm25",
                "sentence-transformers",
                "pymupdf",
                "docx2txt",
                # Unsloth 使用 ModelScope 下载模型时需要
                "modelscope",
            ],
            check=True,
        )
        print("✓ 依赖安装完成")
    except subprocess.CalledProcessError as e:
        print(f"❌ 安装失败: {e}")
        sys.exit(1)
    print()
    
    # 2. 检查streamlit_app.py
    print("✓ 检查文件...")
    if not os.path.exists("streamlit_app.py"):
        print("❌ streamlit_app.py 不存在！")
        sys.exit(1)
    print("✓ streamlit_app.py 已找到")
    print()
    
    # 3. 检查LoRA模型
    print("🔍 检查模型...")
    lora_paths = [
        os.path.join(work_dir, "lora_model"),
        os.path.join(work_dir, "outputs", "checkpoint-60"),
        "/root/autodl-tmp/lora_model",
        "/root/autodl-tmp/outputs/checkpoint-60",
    ]
    for path in lora_paths:
        if os.path.exists(path):
            print(f"✓ 发现LoRA模型: {path}")
            break
    else:
        print("⚠️  未发现LoRA模型，将使用基础模型")
    print()
    
    # 4. 启动Streamlit
    print("=" * 60)
    print("🚀 启动 Streamlit 应用...")
    print("=" * 60)
    print()
    print("📡 访问地址:")
    print("   - 本地: http://localhost:8501")
    print("   - 远程: http://[服务器IP]:8501")
    print()
    print("按 Ctrl+C 停止应用")
    print("-" * 60)
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
             "--server.port", "8501",
             "--server.address", "0.0.0.0",
             "--logger.level", "error"],
            check=True
        )
    except KeyboardInterrupt:
        print("\n\n✓ 应用已停止")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
