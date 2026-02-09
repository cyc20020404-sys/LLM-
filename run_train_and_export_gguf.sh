#!/usr/bin/env bash
# 1) 运行训练，得到 merged_model
# 2) 用 llama.cpp 将 merged_model 转为 GGUF F16，再量化为 Q4_K_M，保存到项目目录

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
OUT_GGUF="$SCRIPT_DIR/my_emotional_bot.Q4_K_M.gguf"
LLAMA_CPP_DIR="$SCRIPT_DIR/llama.cpp"
MERGED_DIR="$SCRIPT_DIR/merged_model"

# 若已有 merged_model，跳过训练，直接做 GGUF 转换
if [ -d "$MERGED_DIR" ] && [ -f "$MERGED_DIR/config.json" ] && ls "$MERGED_DIR"/model*.safetensors 1>/dev/null 2>&1; then
  echo "========== 1/4 训练模型 =========="
  echo "检测到已存在 merged_model，跳过训练，直接导出 GGUF。"
else
  echo "========== 1/4 训练模型 =========="
  python3 train.py
fi

echo ""
echo "========== 2/4 准备 llama.cpp =========="
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMA_CPP_DIR"
fi
cd "$LLAMA_CPP_DIR"
if [ ! -f "convert_hf_to_gguf.py" ]; then
  echo "convert_hf_to_gguf.py 不存在，请检查 clone 是否完整"
  exit 1
fi
pip install -q -r requirements.txt 2>/dev/null || true
pip install -q gguf 2>/dev/null || true

echo ""
echo "========== 3/4 HF → GGUF (F16) =========="
# 中间 F16 约 15G，写到数据盘项目目录（请确保有足够空间）
F16_GGUF="$SCRIPT_DIR/my_emotional_bot_f16.gguf"
python3 convert_hf_to_gguf.py "$SCRIPT_DIR/merged_model" --outfile "$F16_GGUF" --outtype f16

echo ""
echo "========== 4/4 量化 Q4_K_M =========="
if [ ! -f "llama-quantize" ] && [ ! -f "build/bin/llama-quantize" ]; then
  mkdir -p build
  cd build
  cmake .. -DGGML_OPENMP=OFF
  cmake --build . --config Release -j
  cd ..
fi
QUANTIZE="./llama-quantize"
[ -f "build/bin/llama-quantize" ] && QUANTIZE="./build/bin/llama-quantize"
"$QUANTIZE" "$F16_GGUF" "$OUT_GGUF" Q4_K_M

echo ""
echo "========== 完成 =========="
echo "已生成: $OUT_GGUF"
rm -f "$F16_GGUF"
