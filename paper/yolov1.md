YOLOv1(You Only Look Once) 논문은 객체 탐지(Object Detection) 문제를 **단일 회귀 문제(single regression problem)**로 재정의하여 실시간(Real-Time) 성능을 달성한 혁신적인 모델입니다

<img width="698" height="323" alt="image" src="https://github.com/user-attachments/assets/d87499c0-acbc-4053-9f62-5d9bf6657611" />


다음은 YOLOv1의 중요한 내용인 네트워크 구조(Architecture) 및 **학습 방법(Training Method)**에 대한 상세 정리입니다.

1. 네트워크 구조 (Architecture)

YOLO는 이미지에서 바운딩 박스 좌표와 클래스 확률을 하나의 컨볼루션 네트워크로 직접 예측합니다.

1.1. 기본 접근 방식 (Unified Detection)

• 
객체 탐지 프레임워크: 객체 탐지 과정을 여러 단계(영역 제안, 분류, 후처리)로 나누던 기존 방식(R-CNN 등)과 달리, YOLO는 이 모든 단계를 단일 신경망으로 통합했습니다.

• 
그리드 시스템: 입력 이미지를 $S \times S$ 크기의 그리드(Grid)로 나눕니다 (PASCAL VOC의 경우 $S=7$)

    ◦ 객체의 **중심(center)**이 포함된 그리드 셀(Grid Cell)이 해당 객체 탐지를 담당합니다.

• 
출력 예측 (Output Prediction): 각 그리드 셀은 다음 세 가지를 예측합니다:

    ◦ $B$개의 바운딩 박스 및 각 박스의 신뢰도(Confidence Score).
    ◦ $C$개의 조건부 클래스 확률 ($Pr(\text{Class}_i|\text{Object})$).
• 최종 출력 텐서: $S \times S \times (B \times 5 + C)$ 크기의 텐서를 출력합니다.
    ◦ PASCAL VOC의 경우 $S=7, B=2, C=20$이며, 최종 출력은 $7 \times 7 \times 30$ 텐서입니다6.

    ◦ 바운딩 박스 예측 (5개 값): $(x, y, w, h, \text{confidence})$
        ▪ 
$(x, y)$: 바운딩 박스 중심의 좌표로, 해당 그리드 셀의 경계에 대한 상대적인 위치입니다 (0~1 사이 값)7.

        ▪ 
$(w, h)$: 바운딩 박스의 너비와 높이로, 전체 이미지에 대한 상대적인 값입니다8.

        ▪ 
신뢰도(Confidence): $Pr(\text{Object}) \times IOU_{pred}^{\text{truth}}$로 정의되며, 박스에 객체가 포함될 확률과 예측 박스의 정확도를 모두 반영합니다9.


1.2. 네트워크 구조 상세

• 
기반 구조: GoogLeNet 모델에서 영감을 받았습니다10.

• 
레이어 구성: 24개의 컨볼루션 레이어와 **2개의 완전 연결 레이어(Fully Connected Layers)**로 구성되어 있습니다

• 
필터 구조: GoogLeNet의 인셉션 모듈(Inception modules) 대신, $1\times 1$ 축소 레이어 뒤에 $3\times 3$ 컨볼루션 레이어를 사용하는 구조를 사용합니다

• 
입력 해상도: 객체 탐지를 위해 입력 해상도를 $448 \times 448$로 설정합니다


2. 학습 방법 (Training Method)


2.1. 사전 학습 및 전이 학습

• 
사전 학습 (Pretraining): 네트워크의 첫 20개 컨볼루션 레이어를 ImageNet 1000-클래스 분류 데이터셋으로 사전 학습합니다14. 이때 입력 이미지는 $224 \times 224$ 해상도를 사용했습니다

• 
탐지 학습으로 전환: 사전 학습된 모델에 4개의 컨볼루션 레이어와 2개의 완전 연결 레이어를 추가하고, 입력 해상도를 **$448 \times 448$**로 높여 객체 탐지 작업에 맞게 학습합니다


2.2. 손실 함수 (Loss Function)

YOLO는 **제곱 합 오차(Sum-Squared Error, SSE)**를 최적화 목표로 사용합니다. 다만, SSE의 한계를 보완하기 위해 다음과 같은 방법을 도입하고 다중 파트 손실 함수를 사용합니다

1. 좌표 오차 가중치 증가:
    ◦ 객체가 없는 그리드 셀이 많아 모델 학습의 불안정성을 야기하는 문제를 해결하기 위해, 바운딩 박스 좌표 예측 손실에 가중치 $\lambda_{coord}$를 적용하여 증가시킵니다 ($\lambda_{coord}=5$)

2. 객체 없음 오차 가중치 감소:
    ◦ 객체가 없는 박스의 신뢰도 예측 손실에 가중치 $\lambda_{noobj}$를 적용하여 감소시킵니다 ($\lambda_{noobj}=0.5$)

3. 바운딩 박스 크기 오차 조정:
    ◦ 큰 박스와 작은 박스 오차를 동일하게 취급하는 SSE의 문제를 부분적으로 해결하기 위해, 바운딩 박스의 너비($w$)와 높이($h$) 대신 제곱근($\sqrt{w}, \sqrt{h}$)을 예측하여 손실을 계산합니다. 이는 작은 박스의 작은 편차에 더 큰 패널티를 부여합니다

4. 책임 예측자 (Responsible Predictor):
    ◦ 한 그리드 셀이 여러 개의 바운딩 박스를 예측하지만 22, 학습 시에는 실제 객체(Ground Truth)와 IOU(Intersection Over Union)가 가장 높은 예측 박스 하나만 해당 객체 예측에 "책임(responsible)"을 지도록 하여 손실을 계산합니다


2.3. 학습 하이퍼파라미터 및 정규화

• 
활성화 함수 (Activation Function): 마지막 레이어에는 선형 활성화 함수를, 나머지 모든 레이어에는 Leaky ReLU를 사용합니다

• Learning Rate 스케줄:
    ◦ 초반 몇 에포크 동안 $10^{-3}$에서 $10^{-2}$로 학습률을 서서히 증가시킵니다.
    ◦ $10^{-2}$로 75 에포크, $10^{-3}$로 30 에포크, $10^{-4}$로 30 에포크를 학습합니다

• 정규화 (Regularization):
    ◦ 첫 번째 완전 연결 레이어 뒤에 드롭아웃(Dropout) 레이어(rate=0.5)를 사용하여 코어댑테이션(co-adaptation)을 방지합니다

    ◦ **광범위한 데이터 증강(Data Augmentation)**을 적용하여 오버피팅을 방지합니다. 원본 이미지 크기의 최대 20%까지 무작위 스케일링 및 변환, HSV 색 공간에서 노출 및 채도 무작위 조정을 사용합니다27.
