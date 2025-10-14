#!/bin/bash

# 源目录与目标目录
SRC_DIR="/Users/gaobaoqi/Documents/Logseq/assets"
DST_DIR="/Users/gaobaoqi/Documents/LogseqPublish/assets"

# 确保目标目录存在
mkdir -p "$DST_DIR"

# 要复制的文件列表
files=(
"image_1755932262530_0.png"
"image_1755970326969_0.png"
"image_1755971128858_0.png"
"image_1755971642615_0.png"
"image_1755970702936_0.png"
"image_1755971021219_0.png"
"image_1755971327342_0.png"
"image_1755970737567_0.png"
"image_1755970680032_0.png"
"image_1755970894659_0.png"
"image_1755964367366_0.png"
"image_1755963355501_0.png"
"image_1755947804673_0.png"
"image_1755972707348_0.png"
"image_1755972515673_0.png"
"image_1755972547575_0.png"
"image_1755972146678_0.png"
"image_1755973254956_0.png"
"image_1755972685451_0.png"
"image_1755972479087_0.png"
"image_1755972626334_0.png"
"image_1755973325214_0.png"
"image_1755972116418_0.png"
"image_1755972244156_0.png"
"image_1755936544804_0.png"
"image_1755970082005_0.png"
"image_1755970058442_0.png"
"image_1755947389703_0.png"
"image_1755962428478_0.png"
"image_1755962067979_0.png"
"image_1755964216024_0.png"
"image_1755963257266_0.png"
"image_1755947558810_0.png"
"image_1755945603055_0.png"
"image_1755961485811_0.png"
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
