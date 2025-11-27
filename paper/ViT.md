1. Architecture (모델 구조)

<img width="740" height="481" alt="image" src="https://github.com/user-attachments/assets/b2b3bd06-fee7-4c51-bbe5-0d6210547fa9" />

### 1.1 기본 설계 원칙

- **핵심 아이디어**: 이미지를 16×16 패치로 나누고, 이를 단어(token)처럼 처리하여 표준 Transformer 구조에 입력
- **CNN의 귀납적 편향 최소화**: translation equivariance, locality 같은 CNN 고유의 특성을 배제하고 데이터로부터 학습

### 1.2 입력 처리 방식

**패치 임베딩**:

- 입력 이미지 x ∈ R^(H×W×C)를 N개의 패치로 분할
- 각 패치 크기: P×P (보통 16×16 또는 32×32)
- 패치 개수: N = HW/P²
- 평탄화된 패치를 D차원으로 선형 투영

**위치 임베딩**:

- 학습 가능한 1D 위치 임베딩 사용
- 2D-aware 임베딩 대비 성능 차이가 크지 않아 1D 채택
- 패치 임베딩에 더해져서 위치 정보 제공

**[CLS] 토큰**:

- BERT와 유사하게 학습 가능한 classification token을 시퀀스 앞에 추가
- Transformer 출력에서 이 토큰의 표현을 이미지 전체 표현으로 사용

### 1.3 Transformer Encoder

**구조**:

`z'_ℓ = MSA(LN(z_{ℓ-1})) + z_{ℓ-1}    (Multi-head Self-Attention)
z_ℓ = MLP(LN(z'_ℓ)) + z'_ℓ            (MLP)`

- **Layer Normalization**: 각 블록 전에 적용 (pre-norm)
- **Residual Connection**: 각 블록 후에 적용
- **MLP**: 2개 레이어, GELU 활성화 함수
- **L개 레이어 반복** (Base: 12, Large: 24, Huge: 32)

### 1.4 모델 변형

<img width="727" height="122" alt="image (1)" src="https://github.com/user-attachments/assets/ce195278-56ef-4a57-9853-24effb74ce03" />


**명명 규칙**: ViT-L/16 = Large 모델, 16×16 패치 크기

### 1.5 Hybrid Architecture

- CNN feature map을 패치 대신 사용 가능
- ResNet의 중간 feature map을 추출하여 ViT에 입력
- 작은 모델에서는 hybrid가 유리하나 큰 모델에서는 차이 감소

## 2. Method (학습 방법)

### 2.1 Pre-training

**데이터셋**:

- **ImageNet-1k**: 1.3M 이미지, 1k 클래스
- **ImageNet-21k**: 14M 이미지, 21k 클래스
- **JFT-300M**: 303M 고해상도 이미지, 18k 클래스 (구글 내부)

**학습 설정 (JFT-300M)**:

- **Optimizer**: Adam (β₁=0.9, β₂=0.999)
- **Weight Decay**: 0.1 (매우 높은 값)
- **Batch Size**: 4096
- **Learning Rate**:
    - ViT-B: 8×10⁻⁴
    - ViT-L: 4×10⁻⁴
    - ViT-H: 3×10⁻⁴
- **LR Schedule**: Linear warmup (10k steps) + linear decay
- **Epochs**: 7 또는 14
- **Resolution**: 224×224

**정규화 기법**:

- ImageNet 같은 중간 크기 데이터셋에서는 강한 정규화 필요
- Dropout, label smoothing 사용
- ImageNet에서 추가로 gradient clipping (global norm 1)

### 2.2 Fine-tuning

**기본 설정**:

- Pre-trained head 제거, zero-initialized linear layer로 교체
- **Optimizer**: SGD with momentum (0.9)
- **Batch Size**: 512
- **LR Schedule**: Cosine decay
- **Learning Rate**: 그리드 서치 (0.001~0.06)
- **No Weight Decay**
- **Gradient Clipping**: global norm 1

**고해상도 fine-tuning**:

- Pre-training보다 높은 해상도 사용 (일반적으로 384)
- ImageNet 최종 결과: ViT-L/16은 512, ViT-H/14는 518
- 패치 크기는 유지 → 시퀀스 길이 증가
- 위치 임베딩을 2D 보간으로 조정
- Polyak averaging (factor 0.9999) 적용

**VTAB 프로토콜**:

- 모든 19개 task에 동일한 하이퍼파라미터 사용
- Learning Rate: 0.01
- Steps: 2,500
- Resolution: 384×384 (모든 task 공통)

### 2.3 Self-supervised Pre-training (예비 실험)

**Masked Patch Prediction**:

- 50% 패치 손상:
    - 80%: [mask] 토큰으로 교체
    - 10%: 다른 랜덤 패치로 교체
    - 10%: 그대로 유지
- 예측 목표: 각 손상된 패치의 3-bit mean color (512 색상)
- JFT에서 1M steps (약 14 epochs) 학습
- 결과: ImageNet에서 79.9% 달성 (scratch 대비 2% 향상, supervised 대비 4% 낮음)

**결론**: "Large scale training trumps inductive bias" - 충분한 데이터가 있으면 데이터로부터 패턴을 학습하는 것이 더 효과적

## 4. 주요 Ablation Studies

### 4.1 Optimizer 선택 (Table 7)

- ResNet에 Adam 사용이 SGD보다 전반적으로 우수
- JFT 사전학습에서 Adam이 더 안정적

### 4.2 Transformer Shape (Figure 8)

- **Depth scaling이 가장 효과적** (64 layers까지 개선)
- **Width scaling은 효과 제한적**
- **Patch size 감소**(sequence length 증가)가 파라미터 추가 없이 효과적
- **결론**: Depth를 우선시하되 모든 차원 균형있게 스케일

### 4.3 Head Type (Figure 9)

- **[CLS] token vs Global Average Pooling**
- 둘 다 유사한 성능, 단 learning rate가 달라야 함
- [CLS] token이 Transformer 전통을 따름

### 4.4 Positional Embedding (Table 8)

- 1D, 2D, Relative 모두 유사한 성능
- 위치 정보 없으면 큰 성능 저하
- 패치 레벨 입력이라 2D 정보의 이점이 크지 않음

## 5. 핵심 Contributions 및 Insights

### 5.1 주요 기여

1. **최소한의 수정으로 표준 Transformer를 vision에 적용**
2. **대규모 사전학습 시 CNN 능가** 입증
3. **효율적인 학습**: 동등 성능을 더 적은 compute로 달성
4. **스케일링 가능성**: 큰 모델일수록 성능 향상 지속

### 5.2 실용적 권장사항

- **데이터 < 100M**: ResNet 또는 강한 정규화가 필요
- **데이터 > 100M**: ViT가 효율적이고 효과적
- **Fine-tuning**: 고해상도, 위치 임베딩 보간 필수
- **Compute 제약**: Hybrid 모델 고려 (작은 규모에서)

### 5.3 한계 및 향후 과제

- **Self-supervised learning** 아직 supervised 대비 gap 존재
- **Detection, Segmentation** 등 다른 vision task 적용 필요
- **추가 스케일링** 탐구 여지
