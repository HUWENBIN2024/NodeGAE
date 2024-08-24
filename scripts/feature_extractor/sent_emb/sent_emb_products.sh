folder_path="emb"

# 检查文件夹是否存在
if [ ! -d "$folder_path" ]; then
    # 文件夹不存在,创建它
    mkdir -p "$folder_path"
    echo "Folder created: $folder_path"
else
    echo "Folder already exists: $folder_path"
fi
cd src/feature_extractor/sent_emb/
python sent_emb_products.py