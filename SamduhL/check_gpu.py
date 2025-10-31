import torch
print(f"PyTorch version: {torch.__version__}")
is_available = torch.cuda.is_available()
print(f"Is GPU (ROCm) available? ==> {is_available}")
if is_available:
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("\n PyTorch 已经成功连接到您的 GPU")
    print("正在尝试在 GPU 上创建一个张量...")
    try:
        tensor = torch.randn(3, 3).to("cuda")
        print("在 GPU 上的张量:")
        print(tensor)
        print("环境已准备就绪！")
    except Exception as e:
        print(f" !!!在 GPU 上创建张量失败: {e}")
else:
    print("\n !!!PyTorch 无法连接到 GPU")
