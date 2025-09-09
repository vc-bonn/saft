import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return torch.permute(x, self.shape)


class AutoEncoder(nn.Module):
    def __init__(self, opt=False):
        super().__init__()
        self.opt = opt

        self.encoder = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512, affine=False, track_running_stats=False, momentum=None),
        )

        self.decoder = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            Reshape((0, 2, 3, 1)),
            nn.Linear(512, 512),
            Reshape((0, 3, 1, 2)),
            decon(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            decon(256, 128),
            nn.LeakyReLU(negative_slope=0.2),
            decon(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            decon(64, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(negative_slope=0.2),
            decon(32, 9),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(9, 9, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
        )

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def optimiere(self, x):
        output = self.decoder(x)
        return self.activation(output)

    def forward(self, x):
        x = self.manage_inputs(x)
        x = self.encoder(x)
        if self.opt:
            return x
        output = self.decoder(x)

        e = torch.normal(0, 0.2, size=x.shape, device=x.device)
        output_offset = self.decoder(x + e)
        return self.activation(output), self.activation(output_offset)

    def activation(self, output):
        act = torch.sigmoid(output)
        act = self.manage_outputs(act)
        return act

    """
        In: 10d isotropic svbrdf:
            shape: [n,x,y,10] 
            diffuse, roughness, specular: 
                range: [0,1]
                dims: 3, 1, 3 
            normals: 
                range: [-1,1] 
                dims: 3
        Out: 9d isotropic sbnrdf:
        shape: [n,9,x,y]
            diffuse, roughness, specular: 
                range: [0,1]
                dims: 3, 1, 3 
            normals: 
                range: [0,1] 
                dims: 2
    """

    def manage_inputs(self, input):
        diffuse = input[..., 0:3]
        normal = (input[..., 3:6] + 1) / 2  # range [-1,1] to range [0,1]
        rough = input[..., 6:7]
        specular = input[..., 7:10]
        return torch.cat((diffuse, normal[..., 0:2], rough, specular), dim=-1).permute(0, 3, 1, 2)

    def manage_outputs(self, input):
        input = input.permute(0, 2, 3, 1)
        diffuse = input[..., 0:3]
        # normal = (input[..., 3:5] * 2) - 1  # range [0,1] to range [-1,1]
        normal = (input[..., 3:5] / 2 * 2) - 1 / 2  # range [0,1] to range [-1,1]
        rough = input[..., 5:6]
        specular = input[..., 6:9]

        normal_xy = (normal[..., 0] ** 2 + normal[..., 1] ** 2).clamp(min=0, max=1 - 1e-6)
        normal_z = (1 - normal_xy).sqrt()
        normal = torch.stack((normal[..., 0], normal[..., 1], normal_z), -1)
        normalized_normal = normal.div(normal.norm(2.0, -1, keepdim=True))
        return torch.cat((diffuse, normalized_normal, rough, specular), dim=-1)


def decon(in_channels, out_channels, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1)
