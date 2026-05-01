# 방어팀용 attack_index 연동 가이드

## 목적

공격 파트는 공격별 adversarial image와 metadata를 `outputs/attacks/attack_index.csv` 하나로 통합한다. 방어 파트는 공격별 폴더를 직접 탐색하지 말고 이 CSV를 기준으로 입력 이미지를 선택하면 된다.

## 생성 명령

```bash
python src/build_attack_index.py
```

최종 5종 전체 실험 기준 row 수:

```text
fgsm      223
pgd       223
square    223
jsma      223
zoo       223
total    1115
```

## 방어 입력으로 사용할 핵심 컬럼

| Column | Meaning | Usage |
|---|---|---|
| sample_id | 공격 샘플 고유 ID | 방어 결과 join을 위한 필수 키 |
| attack_family | fgsm, pgd, square, jsma, zoo | 공격 계열별 방어 성능 비교 |
| attack | 세부 공격명 | 세부 설정 확인 |
| file | 원본 이미지 경로 | 원본 비교/복구 평가 |
| adv_file | adversarial image 경로 | 방어 입력 이미지 |
| perturbation_file | perturbation 시각화 경로 | 시각 분석용 |
| success | target label로 공격 성공 여부 | 전체 샘플 기준 공격 성공 |
| clean_correct | 원본 이미지가 공격 전 정분류였는지 | 평가 필터 |
| success_on_clean | clean_correct이면서 target 성공인지 | 방어 평가 권장 필터 |
| true_label, true_name | 원본 정답 | 복구 여부 계산 |
| target_label, target_name | 공격 목표 class | target 유지 여부 계산 |
| pred_before_name | 공격 전 예측 | sanity check |
| pred_after_name | 공격 후 예측 | 공격 결과 확인 |
| epsilon, theta, alpha, steps | 공격 파라미터 | 조건별 분석 |
| max_queries, queries_used | black-box query 정보 | Square/ZOO 비용 분석 |
| l0, l2, linf | perturbation 크기 | 공격 강도 분석 |
| time_sec | 공격 생성 시간 | 처리 비용 분석 |
| target_conf_gain | target confidence 증가량 | 공격 효과 분석 |

## 권장 방어 입력 필터

방어 성능은 원본을 모델이 맞혔고 공격도 성공한 샘플을 우선 기준으로 계산하는 것이 좋다.

```python
import pandas as pd

attack_index = pd.read_csv("outputs/attacks/attack_index.csv")
attack_inputs = attack_index[
    (attack_index["clean_correct"] == True) &
    (attack_index["success_on_clean"] == True)
]
```

방어 입력 이미지는 `adv_file`을 사용한다.

```python
for _, row in attack_inputs.iterrows():
    adv_path = row["adv_file"]
    sample_id = row["sample_id"]
```

## 방어 결과 CSV 권장 포맷

방어 파트는 다음 경로에 결과를 저장하는 것을 권장한다.

```text
outputs/defenses/defense_results.csv
```

권장 컬럼:

| Column | Meaning |
|---|---|
| sample_id | attack_index.csv의 sample_id 그대로 유지 |
| attack_family | 공격 계열 |
| attack | 세부 공격명 |
| defense | 방어 기법명 |
| defense_params | JSON 문자열 형태의 방어 파라미터 |
| input_adv_file | 입력 adversarial image |
| defended_file | 방어 적용 후 이미지 경로 |
| pred_before_defense | 방어 전 예측 label |
| pred_after_defense | 방어 후 예측 label |
| target_label | 공격 target label |
| true_label | 원본 true label |
| attack_success_before_defense | 방어 전 target 성공 여부 |
| attack_success_after_defense | 방어 후에도 target 성공인지 여부 |
| recovered | 방어 후 true label로 복구됐는지 여부 |
| target_conf_before_defense | 방어 전 target confidence |
| target_conf_after_defense | 방어 후 target confidence |
| defense_time_sec | 방어 처리 시간 |

## 최소 방어 지표

```text
Defense Success Rate = attack_success_before_defense=True 중 attack_success_after_defense=False 비율
Recovery Rate = attack_success_before_defense=True 중 recovered=True 비율
Target Confidence Drop = target_conf_before_defense - target_conf_after_defense 평균
Defense Time = defense_time_sec 평균
```

## 팀 전달 문장

공격 파트에서 5종 targeted attack 결과를 `attack_index.csv`로 통합해두었다. 방어 파트는 `adv_file`을 입력으로 사용하고, 결과 저장 시 `sample_id`를 반드시 유지하면 된다. 이후 공격 결과와 방어 결과를 `sample_id` 기준으로 join해서 공격별 방어 성공률, 복구율, target confidence 감소량, 처리 시간을 비교할 수 있다.
