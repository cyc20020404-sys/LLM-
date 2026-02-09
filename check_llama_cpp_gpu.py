#!/usr/bin/env python3
"""
检查当前环境中的 llama-cpp-python 是否为 GPU(CUDA) 版，以及运行时是否支持 GPU 卸载。
在项目目录下运行: python check_llama_cpp_gpu.py
"""
import os
import sys
import subprocess

def main():
    print("=" * 60)
    print("llama-cpp-python 安装与 GPU 支持检查")
    print("=" * 60)

    # 1. pip 信息
    print("\n[1] pip 包信息:")
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "show", "llama-cpp-python"],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0 and out.stdout.strip():
            print(out.stdout.strip())
            for line in out.stdout.splitlines():
                if "Location:" in line:
                    loc = line.split(":", 1)[1].strip()
                    pkg_dir = os.path.join(loc, "llama_cpp")
                    lib_dir = os.path.join(pkg_dir, "lib")
                    break
        else:
            print("  未安装 llama-cpp-python")
            return
    except Exception as e:
        print(f"  获取失败: {e}")
        pkg_dir = None
        lib_dir = None

    # 2. 检查 lib 目录里是否有 CUDA 相关 .so（CUDA 版会带 libggml-cuda.so）
    print("\n[2] 安装类型（根据 lib 文件判断）:")
    if lib_dir and os.path.isdir(lib_dir):
        files = os.listdir(lib_dir)
        has_cuda = any("cuda" in f.lower() for f in files)
        print(f"  lib 目录: {lib_dir}")
        print(f"  文件示例: {files[:10]}{'...' if len(files) > 10 else ''}")
        if has_cuda:
            print("  结论: 发现 *cuda* 相关 .so → 应为 **CUDA 版** 安装")
        else:
            print("  结论: 未发现 cuda 相关 .so → 当前为 **CPU 版** 安装（需用 cu124 源重装）")
    else:
        print(f"  未找到 lib 目录: {lib_dir}")

    # 3. 运行时是否支持 GPU 卸载（llama_supports_gpu_offload）
    print("\n[3] 运行时 GPU 卸载支持:")
    try:
        from llama_cpp.llama_cpp import llama_supports_gpu_offload
        ok = bool(llama_supports_gpu_offload())
        if ok:
            print("  llama_supports_gpu_offload() = True → 运行时 **支持 GPU 卸载**")
        else:
            print("  llama_supports_gpu_offload() = False → 运行时 **不支持 GPU 卸载**（推理会用 CPU，常见原因：CUDA .so 加载失败或驱动问题）")
    except Exception as e:
        print(f"  检测异常: {e}")
        import traceback
        traceback.print_exc()

    # 4. 当前 LD_LIBRARY_PATH（若之前设了系统 libstdc++ 可能影响加载）
    print("\n[4] 环境:")
    print(f"  LD_LIBRARY_PATH = {os.environ.get('LD_LIBRARY_PATH', '(未设置)')[:120]}{'...' if len(os.environ.get('LD_LIBRARY_PATH','')) > 120 else ''}")

    print("\n" + "=" * 60)
    print("若上面显示为 CPU 版或不支持 GPU 卸载，请执行:")
    print("  pip uninstall -y llama-cpp-python")
    print("  pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
    print("=" * 60)

if __name__ == "__main__":
    main()
