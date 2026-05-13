
import sys

import numpy as np
import torch
import onnx
import onnxruntime
import matplotlib

from maraboupy import Marabou, MarabouCore

print("Python      :", sys.version.split()[0])
print("torch       :", torch.__version__)
print("onnx        :", onnx.__version__)
print("onnxruntime :", onnxruntime.__version__)
print("numpy       :", np.__version__)
print("matplotlib  :", matplotlib.__version__)

# Marabou 핵심 API가 실제로 노출되는지 확인 (단순 import 가 아닌 attribute 접근)
assert hasattr(Marabou, "read_onnx"), "Marabou.read_onnx not found"
assert hasattr(Marabou, "createOptions"), "Marabou.createOptions not found"
assert hasattr(MarabouCore, "Equation"), "MarabouCore.Equation not found"

print("smoke test OK")
