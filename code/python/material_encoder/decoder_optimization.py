import torch
import math

from material_encoder.AutoEncoiderSmallerFCUp import AutoEncoder
from material_encoder.optimization_functions import OptimizationFunctions
from nvdiffrecmc.render.texture import Texture2D



class Decoder(torch.nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args

        number_tiles = math.floor(self.text_res / (self.tile_res - self.overlap))
        self.texture["texture_resolution"] = number_tiles * (
            self.tile_res - self.overlap
        )

        self._prepare_scaling_grid()
        self._prepare_model()
        self._prepate_latent_code(number_tiles)

        self._set_optimizer()

    @property
    def method_args(self) -> dict:
        return self.args.method_args
    
    @property
    def epochs(self) -> str:
        return self.method_args["optimization"]["epochs"]

    @property
    def texture(self) -> dict:
        return self.method_args["texture"]

    @property
    def overlap(self) -> int:
        return self.texture["overlap"]

    @property
    def text_res(self) -> int:
        return self.texture["texture_resolution"]

    @property
    def tile_res(self) -> int:
        return self.texture["tile_resolution"]

    @property
    def model_path(self) -> str:
        return self.texture["model_path"]

    @property
    def opt_f(self) -> str:
        return self.texture["optimization_function"]

    @property
    def lr(self) -> float:
        return self.texture["lr"]

    @property
    def bsdf(self) -> str:
        return self.texture["bsdf"]

    def _prepare_scaling_grid(self) -> None:
        self.scaling_grid = torch.ones((self.tile_res, self.tile_res, 1)).cuda()
        if self.overlap == 0:
            return

        n2p = torch.linspace(-4, 4, steps=self.overlap).cuda()
        p2n = torch.linspace(4, -4, steps=self.overlap).cuda()

        self.scaling_grid[: self.overlap, ...] = self.scaling_grid[
            : self.overlap, ...
        ] * torch.sigmoid(n2p[:, None, None])
        self.scaling_grid[-self.overlap :, ...] = self.scaling_grid[
            -self.overlap :, ...
        ] * torch.sigmoid(p2n[:, None, None])

        self.scaling_grid[:, : self.overlap, :] = self.scaling_grid[
            :, : self.overlap, :
        ] * torch.sigmoid(n2p[None, :, None])
        self.scaling_grid[:, -self.overlap :, :] = self.scaling_grid[
            :, -self.overlap :, :
        ] * torch.sigmoid(p2n[None, :, None])

    def _prepare_model(self) -> None:
        self.model = AutoEncoder(opt=True)
        # self.model.load_state_dict(torch.load(self.model_path, map_location="cuda:0"))
        self.model = self.model.cuda()
        self.model.decoder = torch.nn.Sequential(
            *(
                list(self.model.decoder.children())[:1]
                + list(self.model.decoder.children())[4:]
            )
        )
        self.model.eval()

    def _prepate_latent_code(self, number_tiles: list[int]) -> None:
        self.latent_codes = []
        for i_x in range(number_tiles):
            for i_y in range(number_tiles):
                latent_code = {}
                # Starting Pixel Positions
                x = i_x * self.tile_res - 1 * i_x * self.overlap - self.overlap
                y = i_y * self.tile_res - 1 * i_y * self.overlap - self.overlap
                # Get all Pixels in x|y
                x = torch.arange(x, x + self.tile_res, dtype=torch.int64)
                y = torch.arange(y, y + self.tile_res, dtype=torch.int64)
                # Wrap around
                x[x > (self.text_res - 1)] -= self.text_res
                y[y > (self.text_res - 1)] -= self.text_res
                # Index for Later
                latent_code["x"], latent_code["y"] = torch.meshgrid(x, y, indexing="ij")

                latent_code["value"] = (
                    torch.rand(
                        1,
                        512,
                        int(self.tile_res / 2**5),
                        int(self.tile_res / 2**5),
                    )
                    * 2
                    - 1
                ).cuda()
                self.latent_codes.append(latent_code)

    def _set_optimizer(self) -> None:
        self.model.decoder.train()
        for name, param in self.model.named_parameters():
            if "decoder" in name:
                param.requires_grad = True
        self.optimizer = OptimizationFunctions(
            self.opt_f, [self.model.decoder.parameters()], [self.lr]
        )
        # lambda1 = lambda epoch: torch.sigmoid(
        #     ((torch.tensor([epoch/self.epochs]).cuda()) + -0.2) * 20.
        # ).item()
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer.optimizer, lr_lambda=[lambda1])

    def forward(self) -> dict:
        output_texture = torch.zeros(self.text_res, self.text_res, 10).cuda()
        textures = self.model.activation(
            self.model.decoder(
                torch.cat([latent["value"] for latent in self.latent_codes])
            )
        )
        if len(self.latent_codes) > 1:

            for i, latent in enumerate(self.latent_codes):
                output_texture[latent["x"], latent["y"]] += (
                    textures[i] * self.scaling_grid
                )
        else:
            return textures
        
        diffuse = torch.cat([output_texture[..., [0, 1, 2]], torch.ones(self.text_res, self.text_res, 1, device = output_texture.device)], dim=-1).unsqueeze(0)
        roughness = output_texture[..., [8]].unsqueeze(0)
        metallic = output_texture[..., [9]].unsqueeze(0)
        normals = output_texture[..., [3, 4, 5]].unsqueeze(0)
        return diffuse, roughness, metallic, None
        # return self._decoder2material(output_texture)

    def _decoder2material(self, texture: torch.Tensor) -> dict:
        if self.bsdf == "pbr":
            return {
                "name": "_estimated_mat",
                "bsdf": "pbr",
                "kd": Texture2D(texture[..., [0, 1, 2]]),
                "ks": Texture2D(texture[..., [7, 8, 9]]),
                "normal": Texture2D(texture[..., [3, 4, 5]]),
            }
        elif self.bsdf == "ct":
            return {
                "name": "_estimated_mat",
                "bsdf": "ct",
                "kd": Texture2D(texture[..., 0:3]),
                "ks": Texture2D(texture[..., 7:10]),
                "r": Texture2D(texture[..., 6:7]),
                "normal": Texture2D(texture[..., [3, 4, 5]]),
            }

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def step(self) -> None:
        self.optimizer.step()
        
    def epoch_update(self):
        pass
        # self.scheduler.step()