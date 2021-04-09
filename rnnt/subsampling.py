
from typing import Tuple
import torch

class BaseSubsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: int, size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)


class Conv2dSubsampling4(BaseSubsampling):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))
        # The right context for every conv layer is computed by:
        # (kernel_size - 1) / 2 * stride  * frame_rate_of_this_layer
        self.subsampling_rate = 4
        # 6 = (3 - 1) / 2 * 2 * 1 + (3 - 1) / 2 * 2 * 2
        self.right_context = 6

    def forward(
            self,
            x: torch.Tensor,
            offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.
            torch.Tensor: positional encoding

        """
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        return x