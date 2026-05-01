import sys

print("=== 한이음 AML 환경 체크 ===")
print(f"Python: {sys.version.split()[0]}")

# PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"  - MPS available: {torch.backends.mps.is_available()}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

# torchvision
try:
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
except ImportError as e:
    print(f"❌ torchvision: {e}")

# ART
try:
    import art
    print(f"ART: {art.__version__}")
except ImportError as e:
    print(f"❌ ART: {e}")

# Foolbox
try:
    import foolbox
    print(f"Foolbox: {foolbox.__version__}")
except ImportError as e:
    print(f"❌ Foolbox: {e}")

# OpenCV
try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

# 간단한 MPS 연산 테스트
try:
    import torch
    if torch.backends.mps.is_available():
        x = torch.randn(1000, 1000, device="mps")
        y = torch.randn(1000, 1000, device="mps")
        z = x @ y  # 행렬곱
        print(f"✓ MPS 연산 테스트 통과 (결과 shape: {z.shape})")
except Exception as e:
    print(f"❌ MPS 연산 실패: {e}")

print("=== 체크 완료 ===")