GPU=0

# parse arguments for output directory
if [ "$1" == "-o" ]; then
    OUTPUT_DIR="$2"
    shift 2
else
    echo "Usage: $0 -o <output_dir>"
    exit 1
fi

CWD=$(pwd)
ABS_OUTPUT_DIR="${CWD}/${OUTPUT_DIR}"

LOC=$(dirname "$0")
cd $LOC


# Preprocess (downscale datasets)
CUDA_VISIBLE_DEVICES=$GPU python downscale_dataset_mip360.py \
--config ../configs/mip360.yml


# Run Gaussian Splatting
# flowers and treehill are not publicly available
# scenes=(bicycle bonsai counter garden kitchen room stump flowers treehill)
scenes=(bicycle bonsai counter garden kitchen room stump)
for scene in "${scenes[@]}";
do
    # Training
    CUDA_VISIBLE_DEVICES=$GPU python ../train.py \
        -m "${ABS_OUTPUT_DIR}/mip360/${scene}" \
        -i "images_vsr" --eval -r 1 \
        --config ../configs/mip360.yml

    # Rendering 
    CUDA_VISIBLE_DEVICES=$GPU python ../render_samename.py \
        -m "${ABS_OUTPUT_DIR}/mip360/${scene}" \
        -i "images_gt" --eval -r 1 --skip_train \
        --config ../configs/mip360.yml

    # Metric
    CUDA_VISIBLE_DEVICES=$GPU python ../metrics.py \
        -m "${ABS_OUTPUT_DIR}/mip360/${scene}"
done

