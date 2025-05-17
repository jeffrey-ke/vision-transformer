import torch
from torch import nn
from torch.nn import Functional


class ViT(nn.Module):
    def __init__(self, num_blocks, proj_dim=1024, patch_dim=16, num_classes=10):
        """
        num_blocks is the Lx number of Norm, MHA, Norm, and MLP blocks in the encoder.
        proj_dim is the dimension the flattened patches are projected into
        patch_dim is the length of one side of a patch.
        """
        ###### Bookkeeping
        num_channels = 3
        self.patch_dim = patch_dim
        self.num_classes = num_classes

        ##### Modules
        self.input_proj = nn.Linear(patch_dim**2 * num_channels, proj_dim)
        self.positional_embedding = PositionalEmbedding()
        self.encoder = TransformerEncoder(L=num_blocks, patch_dim=patch_dim)
        self.cls_head = nn.Linear(proj_dim, num_classes)

    def forward(self, images_batched):
        """
        images_batched: B, 3, H, W
        """
        patches = F.unfold(images_batched,
                           kernel=(self.patch_dim, self.patch_dim),
                           stride=(self.patch_dim, self.patch_dim))
        # patches is shape B, patch_dim, num_patches
        patches_permuted = patches.permute(0, 2, 1)
        # permuted is shape B, num_patches, patch_dim
        projected_patches = self.input_proj(patches_permuted)
        cls_token = nn.Parameter(torch.randn(1, 1, self.patch_dim).expand(B, -1, -1))
        projected_patches = torch.cat([cls_token, projected_patches], dim=1) 

