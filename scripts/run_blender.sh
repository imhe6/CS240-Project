#!/bin/bash

GPU=0
CONFIG="../configs/blender.yml" # 默认值
OUTPUT_DIR="output/blender/"     # 默认值

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--output) OUTPUT_DIR="$2"; shift 2 ;;
        -c|--config) CONFIG="$2"; shift 2 ;;
        -gpu | --gpu) GPU="$2"; shift 2 ;;
        --stage) 
            IFS=',' read -ra STAGE_ARRAY <<< "$2"
            shift 2 ;;
        -h|--help)
            echo "Usage: $0 [-o <output_dir>] [-c <config_path>] [-gpu <CUDA_VISIBLE_DEVICES_ID>] [--stage <train,render,metric>]"
            echo "Options:"
            echo "  -o, --output    Output directory, (default: output/blender)"
            echo "  -c, --config    Path to config file (default: ../configs/blender.yml)"
            echo "  -gpu, --gpu     CUDA_VISIBLE_DEVICES ID (default: 0)"
            echo "  --stage         Comma-separated stages to run (train, render, metric). If not specified, all stages will be run."
            exit 0
            ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory at $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
fi

CWD=$(pwd)
ABS_OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
ABS_CONFIG=$(realpath "$CONFIG") # 转为绝对路径，防止找不到文件

LOC=$(dirname "$0")
cd "$LOC/.." # 切换到项目根目录，确保后续脚本中的相对路径正确

echo "pwd=$(pwd)"
echo "ABS_OUTPUT_DIR=$ABS_OUTPUT_DIR"
echo "ABS_CONFIG=$ABS_CONFIG"
echo "GPU=$GPU"

# Preprocess (downscale datasets)
# CUDA_VISIBLE_DEVICES=$GPU python scripts/downscale_dataset_blender.py \
#     --config "$ABS_CONFIG"

# 指定 stage 参数则仅运行指定步骤，否则运行全部 (train, render, metric)
if [ ${#STAGE_ARRAY[@]} -gt 0 ]; then
    echo "Running stages: ${STAGE_ARRAY[*]}"
else
    echo "No specific stages provided, running all stages (train, render, metric)"
    STAGE_ARRAY=("train" "render" "metric")
fi

# Training
scenes=("chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship")   
for scene in "${scenes[@]}";
do
    for stage in "${STAGE_ARRAY[@]}";
    do
        if [ "$stage" == "train" ]; then
            # Training
            echo "Training on ${scene}..."
            CUDA_VISIBLE_DEVICES=$GPU python train.py \
                -m "${ABS_OUTPUT_DIR}/${scene}" --eval \
                --config "$ABS_CONFIG"
        elif [ "$stage" == "render" ]; then
            # Rendering 
            echo "Rendering ${scene}..."
            CUDA_VISIBLE_DEVICES=$GPU python render_samename.py \
                -m "${ABS_OUTPUT_DIR}/${scene}" \
                --skip_train \
                --config "$ABS_CONFIG"
        elif [ "$stage" == "metric" ]; then
            # Metric
            echo "Calculating metrics for ${scene}..."
            CUDA_VISIBLE_DEVICES=$GPU python metrics.py \
                -m "${ABS_OUTPUT_DIR}/${scene}"
        else
            echo "Unknown stage: $stage. Skipping..."
        fi
    done
    
    # # Training
    # echo "Training on ${scene}..."
    # CUDA_VISIBLE_DEVICES=$GPU python train.py \
    #     -m "${ABS_OUTPUT_DIR}/${scene}" --eval \
    #     --config "$ABS_CONFIG"

    # # Rendering 
    # echo "Rendering ${scene}..."
    # CUDA_VISIBLE_DEVICES=$GPU python render_samename.py \
    #     -m "${ABS_OUTPUT_DIR}/${scene}" \
    #     --skip_train \
    #     --config "$ABS_CONFIG"

    # # Metric
    # echo "Calculating metrics for ${scene}..."
    # CUDA_VISIBLE_DEVICES=$GPU python metrics.py \
    #     -m "${ABS_OUTPUT_DIR}/${scene}"

    echo "--------Finished processing ${scene}--------"
done