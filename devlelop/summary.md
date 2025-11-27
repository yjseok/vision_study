- 안내
    - 자체 제작한 이미지 데이터를 기반으로 영역분할, 3d point clound data의 처리 등의 주요 태스크를 수행하며 데이터 전처리, custom dataset 및 dataloader, 모델 구현, 학습 루프를 구현해 모델의 성능을 검증합니다.
- **통합 문제 요구사항**
    1. RGB 이미지 + Point Cloud 입력
    2. PCD에서 **ground plane 제거(RANSAC)**
    3. Camera intrinsic/extrinsic을 이용해 PCD를 RGB 이미지로 **projection → depth map** 생성
    4. RGB + Depth + Mask 를 하나의 **dataset**으로 구성
    5. UNet 또는 encoder–decoder segmentation 모델 구현
    6. Training Loop (train/val) 직접 구현
    7. mIoU 계산
    8. 최종적으로 segmentation 성능을 검증
    
    즉, **2D + 3D 멀티모달 segmentation** 과제를 온전히 수행할 수 있는 full pipeline이 된다.
    
- **통합 프로젝트에 적합한 Public Dataset**
    - 가장 적합한 데이터셋 = **NYUv2**
        - RGB 이미지 제공
        - Depth 제공 (→ Point cloud 생성 가능)
        - Semantic mask 제공
        - camera intrinsic 제공
    
    NYUv2는 point cloud도 직접 만들어야 하므로 “PCD 전처리, projection”을 테스트하기 아주 좋다.
