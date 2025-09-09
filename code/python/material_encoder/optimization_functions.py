from typing import Any

import torch
# from largesteps.optimize import AdamUniform

# import VectorAdam.vectoradam as va


class OptimizationFunctions:
    """description of class"""

    def __init__(self, f: str, param: list, lrs: list[float],betas=(0.99, 0.9999)) -> None:
        if len(lrs) == 1:
            lrs = [lrs[0] for _ in param]
        params = [{"params": p, "lr": lr} for p, lr in zip(param, lrs)]
        if f == "Adam":
            self.optimizer = torch.optim.Adam(params, betas=betas)
        # elif f == "AdamUniform":
        #     self.optimizer = AdamUniform(params)
        elif f == "vectorAdam":
            raise Exception("Not Refactored")
            """
            for param, ax in zip(params, axis):
                param["axis"] = ax
            self.optimizer = va.VectorAdam(params, betas=(0.99, 0.9999))
            """

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_learning_rate(self):
        values = []
        for param_group in self.optimizer.param_groups:
            values.append(round(param_group["lr"], 5))
        return values

    def __call__(self) -> Any:
        self.step()
        self.zero_grad()
