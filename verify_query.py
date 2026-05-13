"""
단일 Marabou 검증 쿼리를 실행하는 standalone 워커.

test.py 가 subprocess 로 본 스크립트를 호출하여 (sample, ε, alt class) 별로
독립된 프로세스에서 1 회만 solve 한다. 이렇게 하지 않으면 maraboupy 의 C++
측 메모리가 Python GC 만으로 해제되지 않아 sample/ε 를 반복할 때 메모리가
누적되어 OOM 으로 SIGKILL 된다 (확인됨).

사용:
  python verify_query.py --sample IDX --eps EPS --alt ALT \
      [--onnx mnist_cnn.onnx] [--timeout 120] [--out out.json]

stdout 의 마지막 줄에 한 줄짜리 JSON 결과를 출력한다.
exitcode 0 = 정상, 0 아닌 값 = 에러/OOM.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from maraboupy import Marabou
from model import load_pretrained

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
X_MIN = (0.0 - MNIST_MEAN) / MNIST_STD
X_MAX = (1.0 - MNIST_MEAN) / MNIST_STD


def load_sample(idx: int):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])
    ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tf)
    x, y = ds[idx]
    return x.numpy().astype(np.float32), int(y)


def run(args: argparse.Namespace) -> dict:
    x_norm, true_c = load_sample(args.sample)
    x4 = x_norm.reshape(1, 1, 28, 28).astype(np.float32)

    net = Marabou.read_onnx(args.onnx)
    in_vars = np.asarray(net.inputVars[0]).flatten()
    out_vars = np.asarray(net.outputVars[0]).flatten()

    for v, xi in zip(in_vars, x4.flatten()):
        net.setLowerBound(int(v), max(float(xi) - args.eps, X_MIN))
        net.setUpperBound(int(v), min(float(xi) + args.eps, X_MAX))

    # logit_true - logit_alt <= 0 ↔ logit_alt >= logit_true
    net.addInequality(
        [int(out_vars[true_c]), int(out_vars[args.alt])],
        [1.0, -1.0],
        0.0,
        isProperty=False,
    )

    opts = Marabou.createOptions(verbosity=0, timeoutInSeconds=args.timeout)
    t0 = time.time()
    result = net.solve(verbose=False, options=opts)
    elapsed = time.time() - t0

    if isinstance(result, (list, tuple)):
        status = str(result[0])
        vals = result[1] if len(result) > 1 else {}
    else:
        status, vals = str(result), {}

    out = {
        "sample": args.sample,
        "eps": args.eps,
        "alt": args.alt,
        "true_class": true_c,
        "status": status,
        "elapsed": elapsed,
        "adv_image": None,
        "adv_pred": None,
    }
    s_low = status.lower()
    if "sat" in s_low and "unsat" not in s_low and vals:
        try:
            adv = np.array([float(vals[int(v)]) for v in in_vars], dtype=np.float32)
            out["adv_image"] = adv.tolist()
            adv_logits = np.array([float(vals[int(v)]) for v in out_vars], dtype=np.float32)
            out["adv_pred"] = int(np.argmax(adv_logits))
        except KeyError:
            pass
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, required=True)
    p.add_argument("--eps", type=float, required=True)
    p.add_argument("--alt", type=int, required=True)
    p.add_argument("--onnx", default="mnist_cnn.onnx")
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--out", default="")
    args = p.parse_args()

    result = run(args)
    payload = json.dumps(result)
    # 결과 라인 식별을 위해 prefix 사용 (test.py 가 grep)
    print("RESULT_JSON " + payload)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
