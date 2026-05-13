"""
Marabou 검증 디버그용 미니 스크립트.

- 단일 ε, 단일 대안 클래스 1 개만 시도 (disjunction 사용 안 함).
- 어디서 죽는지 단계마다 진행상황을 출력한다.
"""

import sys
import time
from pathlib import Path

# 부모 디렉토리(model.py 가 있는 곳)를 import path 에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

print("[step] importing maraboupy ...", flush=True)
from maraboupy import Marabou, MarabouCore
print("[step] maraboupy imported", flush=True)

from model import load_pretrained

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
X_MIN = (0.0 - MNIST_MEAN) / MNIST_STD
X_MAX = (1.0 - MNIST_MEAN) / MNIST_STD

EPS = 0.005           # 매우 작은 ε
ALT_CLASS = None      # None 이면 runner-up 자동 선택


def load_sample(idx: int):
    tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    x, y = ds[idx]
    return x.numpy().astype(np.float32), int(y)


def main():
    print("[step] loading torch model", flush=True)
    model = load_pretrained("mnist_cnn.pth", device="cpu")

    print("[step] loading sample", flush=True)
    x_norm, true_c = load_sample(0)
    x4 = x_norm.reshape(1, 1, 28, 28).astype(np.float32)

    with torch.no_grad():
        logits = model(torch.from_numpy(x4)).numpy()[0]
    print(f"[step] torch logits: {np.round(logits, 2)}", flush=True)
    print(f"[step] true={true_c}, argmax={int(np.argmax(logits))}", flush=True)

    # runner-up = 두 번째로 큰 logit 의 클래스
    order = np.argsort(logits)
    alt = ALT_CLASS if ALT_CLASS is not None else int(order[-2])
    print(f"[step] alternative class = {alt} (margin = {logits[true_c] - logits[alt]:.3f})", flush=True)

    print("[step] Marabou.read_onnx ...", flush=True)
    t0 = time.time()
    net = Marabou.read_onnx("mnist_cnn.onnx")
    print(f"[step] read_onnx done in {time.time()-t0:.2f}s", flush=True)

    in_vars = np.asarray(net.inputVars[0]).flatten()
    out_vars = np.asarray(net.outputVars[0]).flatten()
    print(f"[step] inputVars={in_vars.size} outputVars={out_vars.size}", flush=True)

    print("[step] setting input bounds ...", flush=True)
    xf = x4.flatten()
    for v, xi in zip(in_vars, xf):
        net.setLowerBound(int(v), max(float(xi) - EPS, X_MIN))
        net.setUpperBound(int(v), min(float(xi) + EPS, X_MAX))

    # 단순 inequality: logit[alt] - logit[true] >= 0  →  -logit[true] + logit[alt] >= 0
    print("[step] adding output inequality (alt - true >= 0) ...", flush=True)
    net.addInequality(
        [int(out_vars[true_c]), int(out_vars[alt])],
        [1.0, -1.0],   # 1*logit_true - 1*logit_alt <= 0  ↔  logit_alt >= logit_true
        0.0,
        isProperty=False,
    )

    print("[step] solving ...", flush=True)
    opts = Marabou.createOptions(verbosity=2, timeoutInSeconds=120)
    t0 = time.time()
    result = net.solve(verbose=True, options=opts)
    print(f"[step] solve returned in {time.time()-t0:.1f}s", flush=True)
    print("[step] result type:", type(result), flush=True)
    print("[step] result[0]:", result[0] if isinstance(result, (list, tuple)) else result, flush=True)

if __name__ == "__main__":
    main()
