#!/bin/bash
# -------------------------------
# 自动提取 Markdown 文件中的图片名并复制到目标目录
# 兼容 macOS 与 Linux
# -------------------------------

# 传入的 Markdown 文件路径
INPUT_FILE="$1"

# 源目录与目标目录
SRC_DIR="/Users/gaobaoqi/Documents/Logseq/assets"
DST_DIR="/Users/gaobaoqi/Documents/LogseqPublish/assets"

# 检查输入
if [[ -z "$INPUT_FILE" ]]; then
    echo "❌ 请提供 Markdown 文件路径，例如："
    echo "   ./copy_images.sh '@From reactive to cognitive%3A brain-inspired spatial intelligence for embodied agents.md'"
    exit 1
fi

if [[ ! -f "./pages/$INPUT_FILE" ]]; then
    echo "❌ 找不到文件: $INPUT_FILE"
    exit 1
fi

# 确保目标目录存在
mkdir -p "$DST_DIR"

# -------------------------------
# 提取所有 image_XXXX_XX.png 文件名并存入 files 数组
# -------------------------------

# macOS BSD sed 兼容写法（不能用 \n，要用反斜杠 + 换行）
files=$(grep -oE 'image_[0-9]+_[0-9]+\.png' "./pages/$INPUT_FILE" \
  | awk '{print "\"" $0 "\""}' \
  | sed '1s/^/files=(\
/; $a\
)')

# 检查是否提取到图片
if [[ -z "$files" ]]; then
    echo "⚠️ 未在文件中找到 image_XXXX_XX.png 格式的图片引用。"
    exit 0
fi

# 使用 eval 将字符串形式的 files 数组转换为真实数组
eval "$files"

# -------------------------------
# 执行复制操作
# -------------------------------
echo "🚀 开始复制图片文件..."
for file in "${files[@]}"; do
    src="$SRC_DIR/$file"
    dst="$DST_DIR/$file"

    if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        echo "✅ 已复制: $file"
    else
        echo "⚠️ 未找到: $file"
    fi
done

echo "🎉 所有文件复制完成！"
