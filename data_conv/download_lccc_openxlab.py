#!/usr/bin/env python3
"""
使用 OpenXLab 下载 LCCC 到 data_conv/LCCC。
需先配置 OpenXLab 鉴权：在终端执行 openxlab login，按提示输入 AK/SK。
（OpenXLab 与 OpenDataLab 同账号体系，可在 https://openxlab.org.cn 用原账号登录后获取 AK/SK）
"""
import os
import sys

def main():
    try:
        from openxlab.dataset import get
    except ImportError:
        print("请先安装: pip install openxlab")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "LCCC")
    os.makedirs(out_dir, exist_ok=True)

    print("正在从 OpenXLab 下载 OpenDataLab/LCCC 到", out_dir, "...")
    get(dataset_repo="OpenDataLab/LCCC", target_path=out_dir)
    print("LCCC 已保存到", out_dir)

if __name__ == "__main__":
    main()
