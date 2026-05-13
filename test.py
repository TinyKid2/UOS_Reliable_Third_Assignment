"""
test.py — Marabou 기반 MNIST_CNN 의 L∞ 견고성(robustness) 검증.

설계 요점 (subprocess 격리):
  - maraboupy 의 C++ 측 메모리가 Python GC 로 해제되지 않아 동일 프로세스에서
    여러 쿼리를 반복하면 RSS 가 누적되어 OOM(SIGKILL) 이 발생함을 확인했다.
  - 따라서 (sample, ε, alt) 단일 쿼리마다 `verify_query.py` 를 별도
    subprocess 로 호출하고, 해당 프로세스 종료와 동시에 메모리를 회수한다.
  - 본 파일은 오케스트레이션 + 결과 집계 + 시각화만 담당한다.

검증 인코딩:
  - 입력 제약: 각 정규화 입력 변수 v_i 에 대해
        max(x_i - ε, X_NORM_MIN) ≤ v_i ≤ min(x_i + ε, X_NORM_MAX)
  - 출력 제약(단일 alt class 단위):
        logit[true] - logit[alt] ≤ 0
    SAT 이면 "alt 클래스가 true 보다 높은 logit 을 가지는 입력 존재" 의미.
  - 각 ε 별로 모든 대안 클래스(9 개) 를 logit 내림차순으로 시도해
    첫 SAT 에서 break.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import load_pretrained


# ──────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

ONNX_PATH = "mnist_cnn.onnx"
# 두 가지 대조적인 샘플:
#   - index  0 (digit 7, margin ≈ 18.75): 매우 confident → 모든 ε 에서 UNSAT 예상
#   - index 95 (digit 4, margin ≈  5.30): 결정경계 근처 → 작은 ε 에서 SAT 가능성
# (scripts/scan_margins.py 로 첫 100 개 샘플의 margin 을 스캔해 선정함.)
SAMPLE_INDICES = [0, 95]
# 메모리 안전 범위. ε = 0.03 에서는 RSS ≈ 15 GB 까지 치솟아 16 GB WSL 에서 OOM.
# 별도 scripts/probe_eps03.py 가 해당 한계를 재현한다.
EPS_LIST = [0.005, 0.01, 0.015, 0.02]
PER_QUERY_TIMEOUT = 120
RESULTS_DIR = Path("results")
DOCS_DIR = Path("docs")
WORKER = "verify_query.py"  # 같은 인터프리터로 호출


# ──────────────────────────────────────────────────────────
# 샘플 로딩 / sanity check (호스트 프로세스 메모리는 작아 안전)
# ──────────────────────────────────────────────────────────
def select_sample(index: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    x, y = ds[index]
    return x.numpy().astype(np.float32), int(y)


def torch_predict(model: torch.nn.Module, x_norm_4d: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(x_norm_4d)).numpy()[0]


# ──────────────────────────────────────────────────────────
# subprocess 호출
# ──────────────────────────────────────────────────────────
def call_worker(sample_idx: int, eps: float, alt: int, timeout: int) -> dict:
    """verify_query.py 를 subprocess 로 호출해 단일 쿼리 결과를 받는다.

    리턴 dict 의 status 가 "OOM" 이면 자식 프로세스가 SIGKILL(보통 OOM) 되었음을 의미.
    """
    cmd = [
        sys.executable, WORKER,
        "--sample", str(sample_idx),
        "--eps", str(eps),
        "--alt", str(alt),
        "--onnx", ONNX_PATH,
        "--timeout", str(timeout),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # 자식 timeout 보다 살짝 여유
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"status": "PROC_TIMEOUT", "elapsed": float(timeout + 60),
                "adv_image": None, "adv_pred": None, "alt": alt}

    if proc.returncode != 0:
        # SIGKILL = -9 = exit -9 (subprocess) or 137 (shell). 보통 OOM.
        status = "OOM" if proc.returncode in (-9, 137) else f"ERR({proc.returncode})"
        return {"status": status, "elapsed": -1.0,
                "adv_image": None, "adv_pred": None, "alt": alt,
                "stderr_tail": proc.stderr[-500:] if proc.stderr else ""}

    # stdout 의 마지막 RESULT_JSON 라인 파싱
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            payload = line[len("RESULT_JSON "):]
    if payload is None:
        return {"status": f"NO_RESULT", "elapsed": -1.0,
                "adv_image": None, "adv_pred": None, "alt": alt,
                "stdout_tail": proc.stdout[-500:]}
    data = json.loads(payload)
    return data


# ──────────────────────────────────────────────────────────
# ε 별 검증 (대안 클래스 순회)
# ──────────────────────────────────────────────────────────
def verify_eps(sample_idx: int, true_c: int, orig_logits: np.ndarray,
               eps: float, timeout: int) -> dict:
    order = [int(c) for c in np.argsort(-orig_logits) if int(c) != true_c]
    per_class = []
    sat_record = None
    saw_timeout = False
    saw_oom = False

    for alt_c in order:
        r = call_worker(sample_idx, eps, alt_c, timeout)
        per_class.append(r)
        status = r.get("status", "").lower()
        print(f"    alt={alt_c}: {r.get('status', '?'):>10}   ({r.get('elapsed', -1):.1f}s)", flush=True)

        if "sat" in status and "unsat" not in status:
            sat_record = r
            break
        if "timeout" in status:
            saw_timeout = True
        if "oom" in status or "err" in status:
            saw_oom = True

    if sat_record is not None:
        overall = "SAT"
    elif saw_oom:
        overall = "OOM"
    elif saw_timeout:
        overall = "TIMEOUT"
    else:
        overall = "UNSAT"

    total_time = sum(max(r.get("elapsed", 0.0), 0.0) for r in per_class)
    return {
        "eps": eps,
        "status": overall,
        "total_time": total_time,
        "per_class": per_class,
        "sat_record": sat_record,
    }


# ──────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────
def visualize_counterexample(x_norm_4d: np.ndarray, adv_norm_4d: np.ndarray,
                             true_c: int, adv_pred: int, eps: float,
                             save_path: Path) -> None:
    def denorm(z: np.ndarray) -> np.ndarray:
        return np.clip(z.squeeze() * MNIST_STD + MNIST_MEAN, 0.0, 1.0)

    orig = denorm(x_norm_4d)
    adv = denorm(adv_norm_4d)
    pert = adv - orig
    lim = max(abs(float(pert.min())), abs(float(pert.max())), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"original\ntrue={true_c}")
    axes[1].imshow(pert, cmap="seismic", vmin=-lim, vmax=lim)
    axes[1].set_title(f"perturbation\nε_norm={eps}")
    axes[2].imshow(adv, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"adversarial\npred={adv_pred}")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"Counterexample (ε_norm={eps})", fontsize=11)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────
def run_one_sample(model: torch.nn.Module, sample_idx: int):
    x_norm, true_c = select_sample(sample_idx)
    x4 = x_norm.reshape(1, 1, 28, 28).astype(np.float32)
    logits = torch_predict(model, x4)
    pred = int(np.argmax(logits))
    runner = float(np.partition(logits, -2)[-2])
    margin = float(logits[true_c] - runner)

    print(f"\n{'#' * 70}", flush=True)
    print(f"# sample {sample_idx}  label={true_c}  pred={pred}  margin={margin:+.3f}", flush=True)
    print(f"{'#' * 70}", flush=True)
    if pred != true_c:
        print(f"[warn] model misclassifies sample {sample_idx} — skipping.")
        return None

    eps_results = []
    for eps in EPS_LIST:
        print(f"\n--- ε = {eps}  (per-query timeout = {PER_QUERY_TIMEOUT}s) ---", flush=True)
        r = verify_eps(sample_idx, true_c, logits, eps, PER_QUERY_TIMEOUT)
        print(f"  → {r['status']}  (cumulative {r['total_time']:.1f}s, queries={len(r['per_class'])})", flush=True)
        if r["sat_record"] is not None and r["sat_record"].get("adv_image"):
            adv_arr = np.array(r["sat_record"]["adv_image"], dtype=np.float32)
            adv4 = adv_arr.reshape(x4.shape)
            png = RESULTS_DIR / f"sample{sample_idx}_eps{eps:.3f}_sat.png"
            visualize_counterexample(x4, adv4, true_c, r["sat_record"]["adv_pred"], eps, png)
            print(f"  saved counterexample → {png}", flush=True)
        eps_results.append(r)

    return {
        "sample_idx": sample_idx,
        "true_class": true_c,
        "pred": pred,
        "margin": margin,
        "logits": logits.tolist(),
        "eps_results": eps_results,
    }


def main():
    if not Path(ONNX_PATH).exists():
        print(f"[error] {ONNX_PATH} not found. Run `python export_onnx.py` first.")
        sys.exit(1)
    if not Path(WORKER).exists():
        print(f"[error] worker script {WORKER} not found.")
        sys.exit(1)

    model = load_pretrained("mnist_cnn.pth", device="cpu")
    print(f"[setup] EPS_LIST = {EPS_LIST}", flush=True)
    print(f"[setup] SAMPLE_INDICES = {SAMPLE_INDICES}", flush=True)
    print(f"[setup] subprocess worker = {WORKER}", flush=True)

    all_results = []
    for idx in SAMPLE_INDICES:
        sr = run_one_sample(model, idx)
        if sr is not None:
            all_results.append(sr)

    # 표 요약 (stdout)
    print("\n" + "=" * 86)
    print("FINAL SUMMARY")
    print("=" * 86)
    header = f"{'sample':>6}  {'true':>4}  {'margin':>8}  {'ε_norm':>8}  {'status':>8}  {'q':>3}  {'time(s)':>8}  {'sat_c':>5}"
    print(header)
    print("-" * 86)
    for sr in all_results:
        for er in sr["eps_results"]:
            sat_c = "-"
            if er["sat_record"] is not None:
                sat_c = str(er["sat_record"].get("alt", "?"))
            print(f"{sr['sample_idx']:>6}  {sr['true_class']:>4}  {sr['margin']:>8.3f}  "
                  f"{er['eps']:>8.4f}  {er['status']:>8}  {len(er['per_class']):>3d}  "
                  f"{er['total_time']:>8.1f}  {sat_c:>5}")

    # 작업 노트 (gitignored)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md = DOCS_DIR / "03_verification_results.md"
    with md.open("w", encoding="utf-8") as f:
        f.write("# 03. Marabou 검증 결과\n\n")
        f.write(f"- 모델: `mnist_cnn.onnx` (input 784 / output 10)\n")
        f.write(f"- 입력 공간: 정규화 MNIST (mean={MNIST_MEAN}, std={MNIST_STD})\n")
        f.write(f"- ε 단위: 정규화 픽셀 (pixel 환산: ε × {MNIST_STD:.4f})\n")
        f.write(f"- per-query timeout: {PER_QUERY_TIMEOUT}s\n")
        f.write(f"- 실행 모드: subprocess 격리 (메모리 누적 방지)\n")
        f.write(f"- 메모리 한계: ε ≥ 0.03 시 RSS ≈ 15 GB → 16 GB WSL 에서 OOM\n\n")

        f.write("## 통합 요약\n\n")
        f.write("| sample | true | margin | ε (norm) | ε (pixel) | status | queries | time (s) | sat class |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for sr in all_results:
            for er in sr["eps_results"]:
                sat_c = "-"
                if er["sat_record"] is not None:
                    sat_c = str(er["sat_record"].get("alt", "?"))
                f.write(
                    f"| {sr['sample_idx']} | {sr['true_class']} | {sr['margin']:.3f} | "
                    f"{er['eps']:.4f} | {er['eps']*MNIST_STD:.4f} | {er['status']} | "
                    f"{len(er['per_class'])} | {er['total_time']:.1f} | {sat_c} |\n"
                )

        f.write("\n## 샘플별 상세\n\n")
        for sr in all_results:
            f.write(f"### sample {sr['sample_idx']} (true={sr['true_class']}, margin={sr['margin']:+.3f})\n\n")
            f.write(f"- logits = {[round(x,2) for x in sr['logits']]}\n\n")
            for er in sr["eps_results"]:
                f.write(f"**ε = {er['eps']}** → **{er['status']}** (총 {er['total_time']:.1f}s)\n\n")
                f.write("| alt class | status | time (s) |\n|---|---|---|\n")
                for q in er["per_class"]:
                    f.write(f"| {q.get('alt', '?')} | {q.get('status', '?')} | {q.get('elapsed', -1):.1f} |\n")
                f.write("\n")
    print(f"\nresults written → {md}")


if __name__ == "__main__":
    main()
