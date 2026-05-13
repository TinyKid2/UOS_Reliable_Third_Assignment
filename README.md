# Reliable and Trustworthy AI — Assignment #3

**Neural Network Verification with Marabou**

본 저장소는 SMT 기반 신경망 검증 도구인 [Marabou](https://github.com/NeuralNetworkVerification/Marabou) 를
사용해, [reliable_AI_First_Assignment](https://github.com/) 에서 학습한 **MNIST CNN** 분류기의
ℓ∞ 견고성(robustness) 을 형식적으로 검증하는 코드와 자료를 담는다.

---

## 1. 디렉토리 구성

```
.
├── model.py             # MNIST_CNN 모델 정의 + 체크포인트 로더
├── mnist_cnn.pth        # First Assignment 의 학습된 가중치
├── export_onnx.py       # PyTorch → ONNX 변환 + 3 단계 sanity check
├── verify_query.py      # 단일 Marabou 쿼리를 실행하는 subprocess 워커
├── test.py              # 검증 오케스트레이터 (subprocess 격리 사용)
├── requirements.txt     # pip 의존성 목록
├── report.pdf           # 1-2 페이지 분석 보고서
├── results/             # SAT 발견 시 counterexample PNG (자동 생성)
└── scripts/             # 보조/디버그 스크립트
    ├── smoke_test.py    # 설치 검증
    ├── scan_margins.py  # 샘플별 logit margin 스캔
    ├── probe_verify.py  # 단일 alt class 검증 디버그
    └── probe_eps03.py   # 메모리 한계 재현용 probe
```

본 저장소에는 포함되지 않는 (gitignore) 자료:

- `Marabou/` — Problem 1 탐색을 위해 clone 한 Marabou repo (필요 시 직접 clone)
- `mnist_cnn.onnx` — `export_onnx.py` 가 생성하는 산출물
- `docs/` — 작업 노트 (설치 로그, 검증 결과 raw 등)
- `.venv/` — Python 가상환경
- `data/` — MNIST 자동 다운로드 디렉토리

---

## 2. 환경 요구사항

### OS / Python

- **OS**: Linux (Ubuntu 권장) 또는 WSL2 Ubuntu. Windows native 는 `maraboupy` 휠이 없어 권장하지 않음.
- **Python**: **3.8 ~ 3.11**. (Marabou 공식 wheel 의 호환 범위. 본 저장소는 3.11.15 로 검증함)

Ubuntu 24.04 의 기본 Python 은 3.12 이므로 별도 환경이 필요하다.

### 권장 설정 — Miniconda

```bash
# Miniconda 설치 후
conda create -y -n marabou python=3.11
conda activate marabou
pip install -r requirements.txt
```

### Marabou repo (Problem 1 탐색용, 선택)

본 검증 코드는 `pip install maraboupy` 만으로 동작한다. 추가로 `resources/`
폴더의 예제/벤치마크를 보고 싶다면 별도로 clone 한다:

```bash
git clone --depth 1 https://github.com/NeuralNetworkVerification/Marabou.git
```

---

## 3. 실행 방법

### 3-1. 설치 검증 (선택)

```bash
python scripts/smoke_test.py
# → maraboupy / torch / onnx / onnxruntime 모두 import 되고 "smoke test OK" 출력
```

### 3-2. PyTorch 모델을 ONNX 로 변환

```bash
python export_onnx.py
```

세 단계 sanity check 를 수행한다:

1. `mnist_cnn.pth` 로딩 → `mnist_cnn.onnx` 생성 (opset 13, 레거시 TorchScript exporter)
2. `onnx.checker` 로 구조 무결성 검증
3. PyTorch eval 출력 vs `onnxruntime` 출력 max abs diff (< 1e-4 기대)
4. `Marabou.read_onnx` 로 그래프 로드 → input/output variable 수 출력

### 3-3. ℓ∞ 견고성 검증 실행

```bash
python test.py
```

내부 동작:

- `SAMPLE_INDICES = [0, 95]` 의 두 MNIST 샘플에 대해
- `EPS_LIST = [0.005, 0.01, 0.015, 0.02]` 의 ℓ∞ 반경(정규화 픽셀 단위)으로
- 각 (sample, ε, alt class) 조합을 **별도 subprocess** 로 `verify_query.py` 호출
- 9 개 대안 클래스를 순회하며 첫 SAT 발견 시 break
- SAT 이면 counterexample 시각화 → `results/sample{idx}_eps{eps}_sat.png`
- 모든 결과를 `docs/03_verification_results.md` 로 저장

기대 실행 시간: 약 8 ~ 12 분 (16 GB RAM, CPU only)

`test.py` 가 `verify_query.py` 를 subprocess 로 호출하는 이유는, `maraboupy`
의 C++ 측 메모리가 같은 Python 프로세스 내에서 반복 호출 시 누적되어 OOM
으로 SIGKILL 되는 현상을 회피하기 위함이다. 자세한 내용은 `report.pdf` 참조.

### 3-4. 단일 쿼리만 디버그 모드로 실행

```bash
python verify_query.py --sample 0 --eps 0.01 --alt 1 --timeout 60
# 마지막 줄에 한 줄짜리 RESULT_JSON {...} 결과
```

---

## 4. 핵심 검증 결과 (요약)

| sample | true | margin | ε (norm) | status |
|---|---|---|---|---|
| 0 | 7 | 18.75 | 0.005 ~ 0.020 | **UNSAT (모두)** ← 견고성 공식 입증 |
| 95 | 4 | 5.30 | 0.005 | 8/9 UNSAT, alt=8 OOM |
| 95 | 4 | 5.30 | 0.010 | 5/9 UNSAT, 4 OOM |
| 95 | 4 | 5.30 | 0.015 ~ 0.020 | 9/9 OOM |

분석/해석은 `report.pdf` 참조.

---

## 5. 트러블슈팅 기록

본 과제 진행 중 만난 주요 이슈 6 가지를 `report.pdf` 의 별도 섹션에 정리.
간단히 요약:

| # | 이슈 | 해결 |
|---|---|---|
| 1 | PEP 668 (Ubuntu 24.04 시스템 pip 차단) | venv 또는 conda 사용 |
| 2 | Python 3.12 에 maraboupy 휠 없음 | Python 3.11 사용 (deadsnakes / miniconda) |
| 3 | torch 2.11 의 dynamo exporter → 미지원 op | `torch.onnx.export(..., dynamo=False)` |
| 4 | maraboupy 의 `Equation` 클래스 혼동 | `MarabouUtils.Equation` 사용 (`MarabouCore.Equation` 은 raw C++ 클래스) |
| 5 | `addDisjunctionConstraint` 메모리 폭발 | 대안 클래스를 sequential per-class inequality 로 풀어 회피 |
| 6 | 동일 프로세스 반복 호출 시 C++ 측 메모리 누적 | subprocess 격리 (`verify_query.py`) |

---

## 6. 라이선스 / 참고문헌

- Marabou: BSD 3-Clause License. <https://github.com/NeuralNetworkVerification/Marabou/blob/master/COPYING>
- Marabou paper: Katz et al., *The Marabou Framework for Verification and Analysis of Deep Neural Networks*, CAV 2019.
- Marabou v2 tool paper: <https://arxiv.org/pdf/2401.14461.pdf>
