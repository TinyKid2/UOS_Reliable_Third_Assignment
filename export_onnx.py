"""
PyTorch MNIST_CNN 체크포인트를 ONNX 로 export 하고 3 단계로 검증한다.

1. PyTorch eval 출력 계산
2. onnxruntime 으로 동일 입력 추론 -> PyTorch 와 max abs diff 비교
3. Marabou.read_onnx 로 로딩 -> 입력/출력 변수 개수 확인

성공 기준:
  - max abs diff < 1e-4 (수치 오차 허용 범위)
  - Marabou 가 network 객체를 정상 반환

산출물:
  - mnist_cnn.onnx (gitignore 대상)
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

from model import load_pretrained


def export_to_onnx(model: torch.nn.Module, onnx_path: str, opset: int = 13) -> None:
    """단일 28x28 그레이스케일 입력에 대해 ONNX 모델을 export 한다.

    구현 메모:
      - torch 2.6+ 의 기본 dynamo exporter 는 `Shape`, `Identity` 등 ONNX op 을
        추가로 삽입하는데, maraboupy 2.0.0 의 ONNXParser 는 이를 지원하지 않는다
        (`NotImplementedError: Operation Shape not implemented`).
      - 따라서 레거시 TorchScript 기반 exporter (`dynamo=False`) 를 사용한다.
      - batch=1 로 고정하면 dynamic_axes 가 필요 없어지고, Marabou 가 기대하는
        단순한 ONNX 그래프가 생성된다.
    """
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )
    print(f"[export] saved -> {onnx_path}  (opset={opset})")


def verify_onnx_structure(onnx_path: str) -> None:
    """onnx.checker 로 모델 구조 무결성 확인."""
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    n_inputs = len(m.graph.input)
    n_outputs = len(m.graph.output)
    n_nodes = len(m.graph.node)
    print(f"[onnx] structure ok. inputs={n_inputs} outputs={n_outputs} nodes={n_nodes}")


def compare_pytorch_vs_ort(model: torch.nn.Module, onnx_path: str, n_samples: int = 5) -> float:
    """무작위 입력으로 PyTorch eval 출력과 onnxruntime 출력 차이 측정.

    리턴:
      n_samples 전체에서의 element-wise max abs diff
    """
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(seed=0)
    max_diff = 0.0
    for i in range(n_samples):
        # MNIST 도메인과 비슷한 [0, 1] 범위 입력
        x = rng.random((1, 1, 28, 28), dtype=np.float32)
        with torch.no_grad():
            y_torch = model(torch.from_numpy(x)).numpy()
        y_ort = sess.run(None, {in_name: x})[0]
        diff = float(np.max(np.abs(y_torch - y_ort)))
        max_diff = max(max_diff, diff)
        print(f"[parity] sample {i}: max|y_torch - y_ort| = {diff:.3e}")
    return max_diff


def verify_marabou_load(onnx_path: str) -> None:
    """Marabou Python API 로 ONNX 모델을 로딩해 구조를 출력한다."""
    # maraboupy 는 WSL 환경에서만 import 가능. 호스트 PowerShell 에서는 import 실패할 수 있다.
    from maraboupy import Marabou

    t0 = time.time()
    net = Marabou.read_onnx(onnx_path)
    elapsed = time.time() - t0

    # API 가 버전마다 약간 달라 try/except 로 양쪽 모두 지원
    try:
        in_vars = net.inputVars[0]
        out_vars = net.outputVars[0]
    except (AttributeError, IndexError):
        in_vars = getattr(net, "inputVars", None)
        out_vars = getattr(net, "outputVars", None)

    n_in = np.asarray(in_vars).size if in_vars is not None else "?"
    n_out = np.asarray(out_vars).size if out_vars is not None else "?"

    print(f"[marabou] read_onnx ok in {elapsed:.2f}s. input_vars={n_in} output_vars={n_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="mnist_cnn.pth")
    parser.add_argument("--onnx", default="mnist_cnn.onnx")
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--skip-marabou", action="store_true",
                        help="maraboupy 가 없는 환경에서 export/parity 만 점검할 때 사용")
    args = parser.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    print(f"[load] checkpoint = {ckpt}")
    model = load_pretrained(str(ckpt), device="cpu")

    export_to_onnx(model, args.onnx, opset=args.opset)
    verify_onnx_structure(args.onnx)
    max_diff = compare_pytorch_vs_ort(model, args.onnx, n_samples=args.samples)
    print(f"[parity] overall max abs diff = {max_diff:.3e}")
    assert max_diff < 1e-4, f"parity check failed (max diff = {max_diff})"

    if args.skip_marabou:
        print("[marabou] skipped by --skip-marabou flag")
    else:
        verify_marabou_load(args.onnx)

    print("\nexport_onnx: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
