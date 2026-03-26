import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.2)

class SudokuDiscriminator(nn.Module):
    def __init__(
        self,
        seq_len: int = 81,
        vocab_size: int = 11,
        hidden_size: int = 128,
        iters: int = 5, 
        forward_dtype: str = "float32",
    ):
        super().__init__()
        assert seq_len == 81
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.iters = iters
        self.forward_dtype = getattr(torch, forward_dtype)
        self._make_discriminator()

    def _make_discriminator(self):
        # Input: puzzle_oh + pred_oh
        in_channels = self.vocab_size * 2
        w = self.hidden_size
        
        self.proj = nn.Conv2d(in_channels, w, kernel_size=3, padding=1)
        self.recur_block = ResidualBlock(w) 
        
        self.classifier = nn.Sequential(
            nn.Linear(w * 9 * 9, w),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(w, 1)
        )

    def _get_nets(self, device):
        if not hasattr(self, 'proj'):
            self._make_discriminator()
            self.to(device)
        return self.proj, self.recur_block, self.classifier

    def forward(self, x_input: torch.Tensor, g_output: torch.Tensor) -> torch.Tensor:
        device = x_input.device
        proj_net, recur_block, classifier_net = self._get_nets(device)

        x_oh = F.one_hot(x_input.long(), num_classes=self.vocab_size).float()
        combined = torch.cat([x_oh, g_output.float()], dim=-1)
        h = combined.transpose(1, 2).view(-1, self.vocab_size * 2, 9, 9)
        
        h = F.leaky_relu(proj_net(h), 0.2)
        
        for _ in range(self.iters):
            h = recur_block(h)
        
        h = h.reshape(h.size(0), -1)
        score = classifier_net(h).squeeze(-1)
        
        return score
