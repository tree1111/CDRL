import torch
import torch.nn as nn


class Stack(nn.Module):
    def __init__(self, channels, height, width):
        super(Stack, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        print(x.shape)
        return x.view(x.size(0), self.channels, self.height, self.width)


# Stride 2 by default
def DeconvBlock(
    in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1, last=False
):
    if not last:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        ),
        nn.Tanh(),
    )


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(3, 256 * 2 * 2)
        self.decoder = nn.Sequential(
            Stack(256, 2, 2),  # Output: (256, 2, 2)
            DeconvBlock(256, 128, 4, stride=2, padding=1, output_padding=0),  # Output: (128, 4, 4)
            DeconvBlock(128, 64, 4, stride=2, padding=1, output_padding=0),  # Output: (64, 8, 8)
            DeconvBlock(64, 28, 4, stride=2, padding=1, output_padding=0),  # Output: (28, 16, 16)
            DeconvBlock(
                28, channels, 3, stride=2, padding=3, output_padding=1, last=True
            ),  # Output: (3, 28, 28)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 256, 2, 2)
        x = self.decoder(x)
        return x


channels = 3
# Example usage:
encoder_output = torch.randn(1, 3)  # Random tensor to represent encoded output
decoder = Decoder()
decoded_image = decoder(encoder_output)
print(decoded_image.shape)  # Should be torch.Size([1, 3, 28, 28])
