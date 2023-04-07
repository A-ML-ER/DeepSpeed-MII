import os
import torch
from transformers import pipeline


def hf_provider(model_path, model_name, task_name, mii_config):
    print(" --- enter hf_provider ---- ")
    if mii_config.load_with_sys_mem:
        device = torch.device("cpu")
        print(" -- device : ")
        print(device)
    else:
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        print(" ---  hf_provider   local_rank---- ")
        print(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(" -- device : ")
        print(device)
    inference_pipeline = pipeline(
        task_name,
        model=model_name,
        device=device,
        framework="pt",
        use_auth_token=mii_config.hf_auth_token,
        torch_dtype=mii_config.dtype,
    )
    print(" ---  hf_provider   success   inference_pipeline ---- ")
    print(inference_pipeline)
    return inference_pipeline
