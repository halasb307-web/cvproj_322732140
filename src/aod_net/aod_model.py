import torch
import torch.nn as nn
import torch.nn.functional as F


class AODNet(nn.Module):
    """
    PyTorch implementation of AOD-Net.

    Input:
        Tensor of shape [B, 3, H, W], with values in [0, 1].

    Output:
        Dehazed tensor of shape [B, 3, H, W].
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))

        concat1 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.conv3(concat1))

        concat2 = torch.cat((x2, x3), dim=1)
        x4 = F.relu(self.conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), dim=1)
        k = F.relu(self.conv5(concat3))

        output = k * x - k + 1.0

        return F.relu(output)


if __name__ == "__main__":
    model = AODNet()
    sample = torch.rand(1, 3, 256, 256)
    result = model(sample)

    print(model)
    print("Input shape:", sample.shape)
    print("Output shape:", result.shape)
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))