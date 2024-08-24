
current_dir_path=$(pwd)
echo "$current_dir_path"

ln -s "$current_dir_path"/src/feature_extractor/autoencoder/vec2text/simteg src/feature_extractor/lm_finetune
ln -s "$current_dir_path"/src/feature_extractor/autoencoder/vec2text/dataset src/feature_extractor/lm_finetune
cd src/feature_extractor/lm_finetune
folder_path="logs"

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    # 文件夹不存在,创建它
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi

folder_path="save_lm_finetune"

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    # 文件夹不存在,创建它
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi

python t5-base_cls_products.py