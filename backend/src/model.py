# # backend/src/model.py

# import torch
# import torch.nn as nn


# # ------------------------------------------------------------
# # Attention Block
# # ------------------------------------------------------------
# class AttentionBlock(nn.Module):
#     def __init__(self, F_g, F_l, F_int):
#         super().__init__()

#         self.W_g = nn.Sequential(
#             nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(F_int)
#         )

#         self.W_x = nn.Sequential(
#             nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(F_int)
#         )

#         self.psi = nn.Sequential(
#             nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(1),
#             nn.Sigmoid()
#         )

#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, g, x):
#         g1 = self.W_g(g)
#         x1 = self.W_x(x)
#         psi = self.relu(g1 + x1)
#         psi = self.psi(psi)
#         return x * psi


# # ------------------------------------------------------------
# # Convolution Block
# # ------------------------------------------------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x):
#         return self.conv(x)


# # ------------------------------------------------------------
# # Attention U-Net
# # ------------------------------------------------------------
# class UNet(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # Encoder
#         self.e1 = ConvBlock(1, 64)
#         self.e2 = ConvBlock(64, 128)
#         self.e3 = ConvBlock(128, 256)
#         self.e4 = ConvBlock(256, 512)

#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#         # Decoder
#         self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#         self.att4 = AttentionBlock(256, 256, 128)
#         self.d4 = ConvBlock(512, 256)

#         self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#         self.att3 = AttentionBlock(128, 128, 64)
#         self.d3 = ConvBlock(256, 128)

#         self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#         self.att2 = AttentionBlock(64, 64, 32)
#         self.d2 = ConvBlock(128, 64)

#         # Output
#         self.out = nn.Conv2d(64, 1, kernel_size=1)

#     def forward(self, x):
#         # Encoder
#         e1 = self.e1(x)
#         e2 = self.e2(self.pool(e1))
#         e3 = self.e3(self.pool(e2))
#         e4 = self.e4(self.pool(e3))

#         # Decoder + Attention
#         d4 = self.up4(e4)
#         e3 = self.att4(d4, e3)
#         d4 = self.d4(torch.cat([d4, e3], dim=1))

#         d3 = self.up3(d4)
#         e2 = self.att3(d3, e2)
#         d3 = self.d3(torch.cat([d3, e2], dim=1))

#         d2 = self.up2(d3)
#         e1 = self.att2(d2, e1)
#         d2 = self.d2(torch.cat([d2, e1], dim=1))

#         return torch.sigmoid(self.out(d2))


import torch
import torch.nn as nn

# ------------------------------------------------------------
# Attention Block
# ------------------------------------------------------------
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ------------------------------------------------------------
# Convolution Block
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ------------------------------------------------------------
# Attention U-Net (MULTI-CLASS)
# ------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, num_classes=4):  # 0=BG, 1=ADC, 2=LCC, 3=SCC
        super().__init__()

        # Encoder
        self.e1 = ConvBlock(1, 64)
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.d4 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.d3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.d2 = ConvBlock(128, 64)

        # 🔑 MULTI-CLASS OUTPUT
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        # Decoder + Attention
        d4 = self.up4(e4)
        e3 = self.att4(d4, e3)
        d4 = self.d4(torch.cat([d4, e3], dim=1))

        d3 = self.up3(d4)
        e2 = self.att3(d3, e2)
        d3 = self.d3(torch.cat([d3, e2], dim=1))

        d2 = self.up2(d3)
        e1 = self.att2(d2, e1)
        d2 = self.d2(torch.cat([d2, e1], dim=1))

        # 🔑 Return logits (softmax will be applied in loss/post-processing)
        return self.out(d2)
