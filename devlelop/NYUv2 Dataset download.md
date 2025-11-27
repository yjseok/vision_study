- **download_nyuv2.sh 로 저장**
    
    ```python
    #!/bin/bash
    
    echo "=============================================="
    echo "NYUv2 Dataset Auto Downloader & Organizer"
    echo "=============================================="
    
    # Base directory
    DATA_DIR="nyuv2"
    mkdir -p $DATA_DIR
    cd $DATA_DIR
    
    echo "[1/5] Downloading NYUv2 images..."
    wget -O nyu_images.zip "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_images.zip"
    
    echo "[2/5] Downloading NYUv2 depth maps..."
    wget -O nyu_depths.zip "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depths.zip"
    
    echo "[3/5] Downloading NYUv2 semantic labels..."
    wget -O nyu_labels.zip "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_labels.zip"
    
    echo "[4/5] Unzipping..."
    unzip -q nyu_images.zip -d images_raw
    unzip -q nyu_depths.zip -d depths_raw
    unzip -q nyu_labels.zip -d labels_raw
    
    echo "[5/5] Organizing folders..."
    mkdir -p rgb depth mask
    
    # move PNG images into rgb/
    mv images_raw/*.png rgb/
    
    # move depth maps (mat -> numpy conversion script needed later, but zip includes PNG depth)
    mv depths_raw/*.png depth/
    
    # move semantic masks
    mv labels_raw/*.png mask/
    
    echo "Cleaning temp folders..."
    rm -rf images_raw depths_raw labels_raw
    
    echo "=============================================="
    echo "NYUv2 READY!"
    echo "Folders created:"
    echo "  nyuv2/"
    echo "    ├── rgb/"
    echo "    ├── depth/"
    echo "    └── mask/"
    echo "=============================================="
    
    ```
    
- 쉘코드 사용
    - chmod +x download_nyuv2.sh
    - ./download_nyuv2.sh
