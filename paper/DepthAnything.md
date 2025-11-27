<img width="1270" height="437" alt="image (2)" src="https://github.com/user-attachments/assets/bd00eaaa-3cac-47b3-b5ab-acf02028d0e9" />

- 연구 목표 : robust하게 작동하는 monocular depth estimation foundation model 구축
    - **기존 문제점**:
        - MiDaS 같은 기존 모델들은 labeled dataset의 coverage 한계로 특정 시나리오에서 성능 저하
        - Depth annotation은 비용이 많이 들고 시간 소모적 (LiDAR, stereo matching, SfM 등)
        - 대규모 depth dataset 구축의 어려움

    - **핵심 아이디어**:
        - **대규모 unlabeled 이미지 활용** (62M개)
        - Labeled 이미지 1.5M + Unlabeled 이미지 62M 결합 학습
        - 단순하지만 효과적인 data scaling-up 전략
- Training Architecture
    - **Encoder**:
        - DINOv2 pre-trained ViT (Vision Transformer)
        - 3가지 크기: ViT-S (24.8M), ViT-B (97.5M), ViT-L (335.3M)
    - **Decoder**:
        - DPT (Dense Prediction Transformer) decoder
        - MiDaS와 동일한 구조 사용
    - **입력 처리**:
        - Training: 짧은 쪽을 518로 resize → 518×518 crop
        - Inference: aspect ratio 유지, 양쪽이 14의 배수가 되도록 조정 (DINOv2 patch size)
- Method (핵심 방법론)
    - 3.1 Two-Stage Training Pipeline
        - **Stage 1: Teacher Model Training (Labeled Images)**
            - 1.5M labeled 이미지로 teacher model T 학습
            - 6개 public datasets 사용:
                - BlendedMVS (115K, stereo)
                - DIML (927K, stereo)
                - HRWSI (20K, stereo)
                - IRS (103K, stereo)
                - MegaDepth (128K, SfM)
                - TartanAir (306K, stereo)
            - 20 epochs 학습
        - **Stage 2: Student Model Training (Labeled + Unlabeled)**
            - Teacher로 unlabeled 이미지에 pseudo label 생성
            - Student model S를 scratch부터 재학습 (fine-tuning 안함)
            - 62M unlabeled 이미지를 1번 sweep
            - Batch 내 labeled:unlabeled = 1:2 비율
    - 3.2 핵심 기술 구성
        - **(1) Affine-Invariant Loss (Multi-dataset Joint Training)**
        서로 다른 dataset의 depth scale/shift 차이를 무시하기 위한 loss:
        `L_l = (1/HW) Σ ρ(d*_i, d_i)`
            - 여기서 ρ는 affine-invariant MAE:
                - Scale과 shift를 align하여 비교
                - `t(d) = median(d)` (translation)
                - `s(d) = (1/HW) Σ |d_i - t(d)|` (scale)
                
                **특징**: Relative depth estimation 가능 (metric이 아닌)
                
        - **(2) Strong Perturbations for Unlabeled Images (핵심!)
        문제 인식**:
            - 단순히 pseudo label로 학습하면 성능 향상 없음
            - Teacher와 student가 같은 architecture/pre-training 공유 → 비슷한 예측
            
            **해결책**: Student에게 더 어려운 optimization target 제공
            **두 가지 perturbation**:
            
            1. **Color Distortions**:
                - Color jittering
                - Gaussian blurring
            2. **Spatial Distortion (CutMix)**:
            
               `u_ab = u_a ⊙ M + u_b ⊙ (1 - M)`
            
            - 두 unlabeled 이미지를 random rectangle mask로 합성
            - 각 region에 대해 별도로 loss 계산 후 weighted average
            - 50% 확률로 적용
            
            **중요**: Teacher는 clean 이미지로 pseudo label 생성, Student는 perturbed 이미지로 학습
            
        - **(3) Semantic-Assisted Perception via Feature Alignment
        초기 시도 (실패)**:
            - Auxiliary semantic segmentation task 추가
            - 4K classes로 unlabeled 이미지에 segmentation label 부여
            - → 성능 향상 없음 (discrete class space의 정보 손실)
            
            **최종 방법 (성공)**:
            
            DINOv2의 rich semantic features를 continuous space에서 직접 align:
            
            `L_feat = 1 - (1/HW) Σ cos(f_i, f'_i)`
            
            - f: Student depth model의 feature
            - f': Frozen DINOv2 encoder의 feature
            - **Tolerance margin α = 0.85**: cosine similarity > α인 pixel은 loss 계산 제외
                - Depth는 같은 object 내에서도 pixel별로 다를 수 있음
                - Semantic encoder는 같은 object에 similar features 생성
                - Margin으로 이 충돌 해결
            
            **부가 효과**: Encoder가 depth + semantic 모두 잘 수행하는 multi-task encoder로 발전
            
        - **(4) Overall Loss**
        `L_total = (L_l + L_u + L_feat) / 3`
        단순 평균 (equal weight)
    - 8. 주요 기여 및 혁신점
        - 8.1 방법론적 혁신
            1. **Large-scale Unlabeled Data의 가치 입증**
                - 기존: labeled data 확보에 집중
                - 본 연구: unlabeled data로 data coverage 확장
            2. **Challenging Optimization Target**
                - Self-training의 한계 극복
                - Strong perturbation으로 extra visual knowledge 학습 강제
            3. **Continuous Semantic Feature Alignment**
                - Discrete segmentation task 대신
                - High-dimensional continuous feature space 활용
        - 8.2 실용적 기여
            1. **Multi-scale Models**
                - ViT-S/B/L 제공
                - Resource-constrained 환경 대응
            2. **Multi-task Encoder**
                - Depth + Semantic 모두 우수
                - Foundation model로서의 가능성
            3. **Better ControlNet**
                - 더 정확한 depth → 더 나은 image synthesis
    - 9. Limitations & Future Work
        
        **현재 한계**:
        
        - 최대 모델 크기: ViT-L
        - Training resolution: 518×518
        
        **향후 계획**:
        
        1. **Model Scaling**: ViT-Giant으로 확장
        2. **Resolution Scaling**: 700+ 또는 1000+ resolution
        3. Better teacher → better pseudo labels
    - 핵심 Takeaways
        1. **Data matters more than architecture**: 새로운 module보다 data scaling-up이 효과적
        2. **Quality of learning > Quantity of data**: Naive self-training은 효과 없음, challenging target 필요
        3. **Feature space > Label space**: Discrete labels보다 continuous features가 rich information 제공
        4. **Foundation models need diversity**: Dataset diversity가 size보다 중요
        
        이 논문은 **simple but effective** 전략으로 robust monocular depth estimation의 새로운 기준을 제시했으며, unlabeled data 활용의 모범 사례를 보여줍니다.
