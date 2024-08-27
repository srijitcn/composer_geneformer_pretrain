import torch
from composer.utils.dist import initialize_dist
import torch.distributed as dist

def run_barrier():

    print("Initializing dist!!")
    initialize_dist(device='gpu')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_tensor = torch.randn(3, 4)

    print(f"On rank {rank}, device: {input_tensor.device}. Before barrier.")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    print(f"On rank {rank}, device: {input_tensor.device}. After barrier.")

run_barrier()
