# 공격 파트 최종 실험 정리 (2026-05-02)

## 완료 범위

LFW-10 전체 test set 223장을 대상으로 targeted adversarial attack 5종을 구현 및 평가했다.

- FGSM: one-step white-box targeted baseline
- PGD: iterative white-box targeted baseline
- Square Attack: query-based black-box targeted baseline
- JSMA 변형: multi-pixel saliency targeted attack
- ZOO-style: finite-difference black-box targeted baseline

실험은 Colab Pro A100 GPU 환경에서 수행했다.

## 모델 및 데이터

- Dataset: LFW deepfunneled
- Task: 10-class face identity classification
- Model: ImageNet pretrained ResNet-50 fine-tuning
- Test samples: 223
- Clean accuracy on test set: 76.23%
- Clean-correct samples: 170 / 223

공격 성공률 비교에는 원본을 모델이 맞힌 샘플만 기준으로 한 `target_success_rate_on_clean`을 핵심 지표로 사용한다.

## 최종 공격 결과

| Attack | Setting | Samples | Clean accuracy | Target ASR on clean | Avg L2 | Avg Linf | Avg time sec | Avg queries |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| FGSM | eps=0.005 | 223 | 76.23% | 27.65% | 1.8920 | 0.0050 | 0.0404 | - |
| PGD | eps=0.030, alpha=0.0030, steps=10 | 223 | 76.23% | 100.00% | 3.5566 | 0.0300 | 0.2181 | - |
| Square | eps=0.050, queries=300 | 223 | 76.23% | 59.41% | 10.7586 | 0.0500 | 1.3275 | 190.77 |
| JSMA 변형 | theta=0.050, steps=20, k=200 | 223 | 76.23% | 99.41% | 1.9725 | 0.0491 | 0.2698 | - |
| ZOO-style | eps=0.050, queries=2000 | 223 | 76.23% | 7.06% | 0.5845 | 0.0340 | 0.6175 | 1891.51 |

## 해석

White-box 계열에서는 PGD가 100.00%, JSMA 변형이 99.41%의 targeted ASR on clean을 달성했다. FGSM은 가장 빠르지만 targeted setting에서는 27.65%로 성공률이 낮아 one-step baseline으로 보는 것이 적절하다.

Black-box 계열에서는 Square Attack이 평균 190.77 query에서 59.41%의 성공률을 보였다. 반면 ZOO-style finite-difference attack은 평균 1891.51 query를 사용했음에도 7.06% 성공률에 그쳐, 현재 설정에서는 Square보다 query 효율이 낮은 baseline으로 확인됐다.

## 산출물

Colab output 기준:

```text
outputs/attacks/compact_attack_summary.csv
outputs/attacks/face_attack_summary.csv
outputs/attacks/attack_index.csv
outputs/attacks/representative_samples.csv
outputs/attacks/figures/
outputs/attack_panels/representatives/
```

Google Drive 백업 위치:

```text
/content/drive/MyDrive/hanium-aml/results/
```

최종 attack index 규모:

```text
fgsm      223
pgd       223
square    223
jsma      223
zoo       223
total    1115
```

## 보고용 요약 문장

공격 파트에서는 ResNet-50 기반 얼굴인식 모델을 대상으로 FGSM, PGD, Square Attack, JSMA 변형, ZOO-style finite-difference attack을 모두 targeted 설정으로 구현 및 평가하였다. LFW-10 전체 test set 223장 기준으로 실험한 결과, white-box 계열에서는 PGD가 100.00%, JSMA 변형이 99.41%의 targeted ASR on clean을 달성했다. Black-box 계열에서는 Square Attack이 평균 190.77 query에서 59.41%의 성공률을 보인 반면, ZOO-style attack은 평균 1891.51 query를 사용했음에도 7.06%의 성공률에 그쳐 query 효율성이 낮은 baseline으로 확인되었다.
