"""
MNIST CNN 모델 정의.

reliable_AI_First_Assignment 에서 학습한 모델을 그대로 재사용한다.
Marabou로 검증할 때는 ONNX로 export 한 뒤 verifier 에 로딩한다.

구조 (입력 1x28x28):
  Conv2d(1, 32, 3) -> ReLU -> MaxPool(2)        # 26x26 -> 13x13
  Conv2d(32, 64, 3) -> ReLU -> MaxPool(2)       # 11x11 -> 5x5
  Flatten -> Linear(1600, 128) -> ReLU
  Dropout(0.5)   (eval 모드에서는 identity)
  Linear(128, 10)  (logits)

Marabou 친화적인 이유:
  - 활성함수가 ReLU 뿐 (Marabou의 1순위 지원)
  - BatchNorm 없음 -> ONNX export 후 추가 fold 필요 없음
  - MaxPool 은 Marabou에서 piecewise-linear 로 처리 가능
"""

import torch
import torch.nn as nn


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.block2(self.block1(x)))


def load_pretrained(checkpoint_path: str = "mnist_cnn.pth",
                    device: str = "cpu") -> MNIST_CNN:
    """저장된 체크포인트로부터 모델을 로딩해 eval 모드로 반환."""
    model = MNIST_CNN().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
