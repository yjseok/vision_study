1. `mat_to_depth_converter.py` — `.mat` 파일에서 depth, rgb, label(40-class) 추출 → `.npy`/`.png`/`.npy_labels`로 저장
2. `make_splits_and_lists.py` — (옵션) `splits.mat`(공식) 사용해서 train/test split 파일 생성. 파일 없으면 랜덤 split 생성
3. `normalize_depth.py` — PNG/16bit/uint16 또는 `.npy` 깊이 파일을 meter 단위 float32로 정규화 & hole filling (선택)
4. `remap_nyu40_to_13.py` — 40→13 클래스 리맵 스크립트 (공식 매핑 파일이 있으면 자동 로드, 없으면 placeholder 사용법 제공)
5. `augmentation_pipeline.py` — 이미지+depth+mask에 동일하게 적용 가능한 augmentation 클래스 (PyTorch `transforms`like)

> 사용 순서(권장)
> 
> 1. `mat_to_depth_converter.py`로 `.mat` → 원시 파일로 변환
> 2. `normalize_depth.py`로 원하는 포맷(메터 단위 .npy 또는 16bit PNG)으로 정규화
> 3. `remap_nyu40_to_13.py`로 라벨 리맵(필요시)
> 4. `make_splits_and_lists.py`로 train/val 리스트 생성
> 5. 학습 시 `augmentation_pipeline.py`의 `Augmentor`를 Dataset에서 사용
