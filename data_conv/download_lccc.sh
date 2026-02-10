#!/bin/bash
# 使用 OpenXLab 将 LCCC 下载到 data_conv（需先 openxlab login 或设置 OPENXLAB_AK/OPENXLAB_SK）
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"
mkdir -p data_conv
openxlab dataset get -r OpenDataLab/LCCC -t "$(pwd)/data_conv"
echo "LCCC 已下载到 data_conv 下，请查看 data_conv/ 目录。"
