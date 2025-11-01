#!/bin/bash

# 源目录与目标目录
SRC_DIR="/Users/gaobaoqi/Documents/Logseq/assets"
DST_DIR="/Users/gaobaoqi/Documents/LogseqPublish/assets"

# 确保目标目录存在
mkdir -p "$DST_DIR"

# 要复制的文件列表
files=(
"image_1761754049084_0.png"
"image_1761753902764_0.png"
"image_1761754162525_0.png"
"image_1761754990048_0.png"
"image_1761755461942_0.png"
"image_1761814563726_0.png"
"image_1761815226762_0.png"
"image_1761827205049_0.png"
"image_1761827933384_0.png"
"image_1761827751658_0.png"
"image_1761828030241_0.png"
"image_1761828355949_0.png"
"image_1761829807108_0.png"
"image_1761829843250_0.png"
"image_1761829886341_0.png"
"image_1761830008846_0.png"
"image_1761830037166_0.png"
)

# 执行复制操作
echo "开始复制图片文件..."

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
