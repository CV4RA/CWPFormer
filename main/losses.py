import torch.nn as nn

def get_loss_fn(loss_name="cross_entropy"):
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_name == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Loss function '{loss_name}' not supported.")
