import torch
import subprocess
import sys


def diagnose_gpu_issue():
    """诊断GPU识别问题"""
    print("=" * 50)
    print("GPU诊断信息")
    print("=" * 50)

    # 基本PyTorch信息
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA能力: {torch.cuda.get_device_capability(i)}")
            print(f"  总内存: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("❌ PyTorch无法检测到CUDA")

    print("\n" + "=" * 50)
    print("系统CUDA信息")
    print("=" * 50)

    # 检查系统CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("NVCC (CUDA编译器) 已安装:")
            print(result.stdout)
        else:
            print("❌ NVCC未找到")
    except FileNotFoundError:
        print("❌ NVCC未安装")

    # 检查PyTorch的CUDA版本
    if hasattr(torch.version, 'cuda'):
        print(f"PyTorch编译的CUDA版本: {torch.version.cuda}")

    print("\n" + "=" * 50)
    print("解决方案")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("1. 可能需要重新安装GPU版本的PyTorch")
        print("2. 运行: pip uninstall torch torchvision torchaudio")
        print(
            "3. 然后安装GPU版本: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")


if __name__ == "__main__":
    diagnose_gpu_issue()