import torch
from prettytable import PrettyTable


def display_model_params(model: torch.tensor):
    table = PrettyTable(["Network Component", "# Parameters"])
    num_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        table.add_row([name, parameter.numel()])
        num_params += parameter.numel()
    print(table)
    print(f"Total Trainable Paramseters: {num_params}")
