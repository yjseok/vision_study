- Swin-Unet - O
    
    <img width="502" height="619" alt="스크린샷 2025-11-29 오후 5 30 08" src="https://github.com/user-attachments/assets/20ee13d8-2b45-4b0e-bfaf-c7d28cf46143" />

    
    ## 💡 Swin-Unet 개요
    
    Swin-Unet은 의료 영상 분할(Medical Image Segmentation)을 위해 제안된 **U-Net과 유사한 순수 Transformer 기반의 인코더-디코더 신경망**입니다. 기존 CNN 기반 모델이 컨볼루션 연산의 **지역성(locality)**으로 인해 전역적(global)이고 장거리(long-range)의 의미론적 정보 상호작용을 학습하는 데 어려움이 있다는 문제를 해결하기 위해 고안되었습니다.
    
    **🛠️ 훈련 아키텍처 (Training Architecture)**
    
    Swin-Unet은 **Encoder, Bottleneck, Decoder, Skip Connection**으로 구성된 대칭적인 U자형 구조를 가집니다. Swin-Unet의 기본 구성 요소는 **Swin Transformer 블록**입니다
    
    **1. 인코더 (Encoder)**
    
    • **입력 처리:** 입력 의료 영상은 4 x 4 크기의 **겹치지 않는 이미지 패치**로 분할되며, 각 패치는 **토큰**으로 간주됩니다. 패치 분할 후, **선형 임베딩 계층(Linear Embedding Layer)**을 적용하여 특성 차원(feature dimension)을 임의의 차원 C로 투영합니다.
    
    • **계층적 특징 추출:** 인코더는 **Swin Transformer 블록**과 **패치 병합 계층(Patch Merging Layer)**을 반복적으로 사용하여 계층적 특징 표현(hierarchical feature representations)을 생성합니다.
    
    • **패치 병합 계층 (Patch Merging Layer):** **다운 샘플링**과 **차원 증가**를 담당합니다.
    
        ◦ 입력 패치를 4개 부분으로 나누어 연결(concatenate)하여 해상도를 **2배 다운 샘플링**합니다.
    
        ◦ 연결 연산으로 인해 4배 증가한 특성 차원을 다시 **선형 계층(Linear Layer)**을 적용하여 원래 차원의 2배로 맞춥니다.
    
    • **Swin Transformer 블록:** **계층적 Swin Transformer와 이동 창(shifted windows)**을 사용하여 문맥 특징을 추출합니다. 이를 통해 **지역(local)에서 전역(global)으로의 자기 주의(self-attention)**가 구현됩니다.
    
    **2. 병목 (Bottleneck)**
    
    • 병목은 **두 개의 연속적인 Swin Transformer 블록**으로 구성되어 깊은 특징 표현을 학습합니다.
    
    • 특징의 차원과 해상도는 **변화 없이 유지**됩니다.
    
    **3. 디코더 (Decoder)**
    
    • 인코더에 대응하는 **대칭적인 구조**로, **Swin Transformer 블록**과 **패치 확장 계층(Patch Expanding Layer)**으로 구성됩니다.
    
    • **패치 확장 계층 (Patch Expanding Layer):** **업 샘플링**을 수행하며, 컨볼루션이나 보간법을 사용하지 않습니다.
    
        ◦ 인접한 차원의 특징 맵을 해상도 **2배 업 샘플링**을 가진 더 큰 특징 맵으로 모양을 변경하고, 이에 따라 특징 차원을 원래 차원의 절반으로 줄입니다.
    
        ◦ 예를 들어, 첫 번째 패치 확장 계층에서는 선형 계층을 적용하여 입력 특징의 차원을 2배 증가시킨 후, **재배열(rearrange) 연산**을 사용하여 해상도를 2배 확장하고 차원을 1/4로 줄입니다.
    
    • **출력:** 마지막 패치 확장 계층은 해상도를 입력 해상도 W x H 로 복원하기 위해 **4배 업 샘플링**을 수행하며, 이후 **선형 투영 계층(Linear Projection Layer)**을 적용하여 픽셀 수준의 분할 예측을 출력합니다.
    
    **4. 스킵 연결 (Skip Connection)**
    
    • U-Net과 유사하게, 스킵 연결은 **인코더의 다중 스케일 특징**과 **디코더의 업 샘플링된 특징**을 융합하는 데 사용됩니다.
    
    • 이는 다운 샘플링으로 인한 **공간 정보의 손실을 완화**하는 역할을 합니다.
    
    • 인코더의 얕은 특징과 디코더의 깊은 특징을 **연결(concatenate)**한 후, 선형 계층을 통해 차원을 업 샘플링된 특징의 차원과 동일하게 유지합니다.
    
    • 실험 결과, 스킵 연결의 개수가 증가할수록 모델 성능이 향상되었으며, 최종 모델에서는 **3개의 스킵 연결**(1/4, 1/8, 1/16 해상도 스케일)을 사용했습니다.
    
    **🧪 방법 및 구현 상세 (Method and Implementation Details)**
    
    **1. Swin Transformer 블록**
    
    • Swin Transformer 블록은 **이동 창 기반의 자기 주의(shifted windows)**를 기반으로 합니다.
    
    • 각 Swin Transformer 블록은 **LayerNorm (LN) 계층**, **다중 헤드 자기 주의 모듈**, **잔차 연결(residual connection)**, 그리고 **GELU 비선형성을 갖는 2계층 MLP**로 구성됩니다.
    
    • 연속적인 두 개의 Swin Transformer 블록은 각각 **창 기반(Window-based, W-MSA)** 및 **이동 창 기반(Shifted Window-based, SW-MSA)**의 다중 헤드 자기 주의 모듈을 적용합니다.
    
    **2. 훈련 상세**
    
    • **구현 환경:** Python 3.6 및 Pytorch 1.7.0 기반.
    
    • **데이터 증강:** 뒤집기(flips) 및 회전(rotations) 사용.
    
    • **입력/패치 크기:** 입력 이미지 크기는 224 x 224, 패치 크기는 4로 설정.
    
    • **초기화:** ImageNet에서 사전 훈련된 가중치를 사용하여 모델 매개변수를 초기화.
    
    • **옵티마이저:** **SGD 옵티마이저** (모멘텀 0.9, 가중치 감소 10^{-4}), 배치 크기 2430.
    
    ## 📈 평가 (Evaluation)
    
    ### 1. 데이터셋
    
    - **Synapse 다중 장기 분할 데이터셋 (Synapse):** 30개 케이스의 3779개 복부 CT 이미지. 8개 복부 장기(대동맥, 담낭, 왼쪽 신장, 오른쪽 신장, 간, 췌장, 비장, 위) 분할에 사용.
    - **자동 심장 진단 챌린지 데이터셋 (ACDC):** MRI 스캐너를 사용하여 수집. 좌심실(LV), 우심실(RV), 심근(MYO) 분할에 사용.
    
    ### 2. 평가 지표
    
    - **평균 Dice-Similarity Coefficient (DSC):** 분할 정확도.
    - **평균 Hausdorff Distance (HD):** 경계 예측 정확도.

**4. 절제 연구 (Ablation Study)**

• **업 샘플링 방법 영향:**
    ◦ **Patch Expanding Layer**를 사용했을 때 (DSC: 79.13%)가 Bilinear interpolation (76.15%)이나 Transposed convolution (77.63%)을 사용했을 때보다 **더 나은 분할 정확도**를 얻었습니다.

• **스킵 연결 개수 영향:**
    ◦ 스킵 연결 개수가 0개(DSC: 72.46%)에서 3개(DSC: 79.13%)로 증가함에 따라 **분할 성능이 지속적으로 증가**했습니다.

• **입력 크기 영향:**
    ◦ 입력 크기를 224 x 224에서 384 x 384로 증가시키면 성능이 약간 향상되지만(DSC: 79.13% 81.12%), **계산 부하도 크게 증가**합니다. 논문에서는 효율성을 위해 224 x 224를 채택했습니다.

Swin-Unet은 순수 Transformer 기반으로 전역 및 장거리 의미론적 정보 상호작용을 더 잘 학습하여, 특히 경계 예측에서 뛰어난 성능을 보임을 시연했습니다.
