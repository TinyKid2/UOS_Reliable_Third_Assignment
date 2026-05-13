"""ε=0.03 단일 alt class 검증으로 어디서 죽는지 디버그."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from maraboupy import Marabou
from model import load_pretrained

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
X_MIN = (0.0 - MNIST_MEAN) / MNIST_STD
X_MAX = (1.0 - MNIST_MEAN) / MNIST_STD

EPS = 0.005
ALT = 9   # 임의로 가까운 클래스 (sample 95 의 logits 분포 기준 변경 가능)
SAMPLE_IDX = 95

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tf)
x, y = ds[SAMPLE_IDX]
x4 = x.numpy().reshape(1, 1, 28, 28).astype(np.float32)
true_c = int(y)
print(f"sample true={true_c}", flush=True)

net = Marabou.read_onnx("mnist_cnn.onnx")
in_vars = np.asarray(net.inputVars[0]).flatten()
out_vars = np.asarray(net.outputVars[0]).flatten()

xf = x4.flatten()
for v, xi in zip(in_vars, xf):
    net.setLowerBound(int(v), max(float(xi) - EPS, X_MIN))
    net.setUpperBound(int(v), min(float(xi) + EPS, X_MAX))

net.addInequality([int(out_vars[true_c]), int(out_vars[ALT])], [1.0, -1.0], 0.0)

opts = Marabou.createOptions(verbosity=2, timeoutInSeconds=180)
print(f"[probe] solving ε={EPS}, alt={ALT} ...", flush=True)
t0 = time.time()
res = net.solve(verbose=True, options=opts)
print(f"[probe] done in {time.time()-t0:.1f}s", flush=True)
print(f"[probe] status: {res[0] if isinstance(res,(list,tuple)) else res}", flush=True)
