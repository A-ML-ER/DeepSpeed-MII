import mii

mii_configs = {'tensor_parallel': 1}

# gpt2
name = "microsoft/DialoGPT-large"

print(f"Deploying {name}...")

mii.deploy(task='conversational', model=name, deployment_name=name + "_deployment")
