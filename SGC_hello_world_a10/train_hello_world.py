import torch

if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    a = torch.randn(1000, 1000, device="cuda")
    b = torch.randn(1000, 1000, device="cuda")
    c = torch.matmul(a, b)

    print(f"Matmul result shape: {c.shape}")
    print("CUDA is working!")
else:
    print("CUDA not available")