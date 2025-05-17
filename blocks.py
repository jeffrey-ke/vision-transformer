import torch
from torch import nn
from torch.nn import Functional

class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, ):
        super().__init__()
        self.embedding = nn.Embedding(len=embed_dim)

    def forward(self, tokens):
        # tokens will be shape B, S, dim
        _,num_tokens,_ = tokens.shape
        positions = self.embedding(torch.arange(num_tokens))
        return tokens + positions

class ViT(nn.Module):
    def __init__(self, num_blocks, proj_dim=1024, patch_dim=16, num_classes=10):
        """
        num_blocks is the Lx number of Norm, MHA, Norm, and MLP blocks in the encoder.
        proj_dim is the dimension the flattened patches are projected into
        patch_dim is the length of one side of a patch.
        """
        super().__init__()
        ###### Bookkeeping
        num_channels = 3
        self.patch_dim = patch_dim
        self.num_classes = num_classes

        ##### Modules
        self.input_proj = nn.Linear(patch_dim**2 * num_channels, proj_dim)
        self.positional_embedding = PositionalEmbedding(embed_dim=proj_dim)
        self.encoder = TransformerEncoder(L=num_blocks, patch_dim=patch_dim)
        self.cls_head = nn.Sequential([nn.Linear(proj_dim, proj_dim), nn.ReLU(), nn.Linear(proj_dim, num_classes)])
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_dim).expand(B, -1, -1))

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
        projected_patches = torch.cat([cls_token, projected_patches], dim=1) 
        patches_embedded = self.positional_embedding(projected_patches)
        tokens = self.encoder(patches_embedded)
        cls_tokens = tokens[:,0,:] #B, patch_dim
        logits = self.cls_head(cls_tokens) #B, num_classes
        return torch.argmax(logits, dim=-1) #B, 


