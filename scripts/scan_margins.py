"""처음 N 개 테스트 샘플의 runner-up 대비 logit margin 을 스캔하여
견고성 검증이 어려울 (마진이 작은) 후보 샘플을 추천한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from model import load_pretrained

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
SCAN_N = 100

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
ds = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=tf)

model = load_pretrained("mnist_cnn.pth", device="cpu")
results = []
with torch.no_grad():
    for i in range(SCAN_N):
        x, y = ds[i]
        logits = model(x.unsqueeze(0)).numpy()[0]
        pred = int(np.argmax(logits))
        if pred != int(y):
            continue
        margin = float(logits[pred] - np.partition(logits, -2)[-2])
        results.append((margin, i, int(y), pred))

# margin 오름차순 (작은 margin = 잘 견고하지 않을 가능성 큼)
results.sort()
print(f"scanned {SCAN_N} samples, {len(results)} correctly classified")
print(f"\n{'idx':>5}  {'label':>5}  {'pred':>5}  {'margin':>8}")
for margin, idx, lbl, pred in results[:15]:
    print(f"{idx:>5}  {lbl:>5}  {pred:>5}  {margin:>8.3f}")
