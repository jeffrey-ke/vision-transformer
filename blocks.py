import math

import torch
from torch import nn
from torch.nn import Functional as F

"""
Another thing that's useful to ask: 
    How much data did the ViT need to get competitive performance 
    with the best ResNet? Is is true that transformers need more
    data?


What are the best community implementations of
these modules? Look them up, don't handroll 
your own. i.e. does xformers do this better, 
does Meta have their own implementation, 
would I ever use a ViT from scratch or
would I always use something like DINO?



Measure of success: you train this on all of ImageNet and get all your pretty graphs
Remember that a failure is more interesting than a 
success here.

"""
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_k, d_v):
        super().__init__()
        ### Modules
        self.key_proj = nn.Linear(d_model, num_heads * d_k)
        self.query_proj = nn.Linear(d_model, num_heads * d_k)
        self.value_proj = nn.Linear(d_model, num_heads * d_v)

        self.output_proj = nn.Linear(num_heads * d_v, d_model)

        ### State variables
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v

    def forward(self, keys, queries, values):
        """
        Separating keys, queries, values so I can do cross attention if needed.
        """
        assert(keys.shape[-1] == self.d_k and values.shape[-1] == self.d_v)
        batches, num_tokens, *_ = tokens.shape
        num_heads = self.num_heads
        d_k = self.d_k
        d_v = self.d_v

        keys = (
                self.key_proj(keys)
                .view(batches, num_tokens, num_heads, d_k)
                .transpose(1, 2)
        ) #b,h,s,k
        queries = (
                self.query_proj(queries)
                .view(batches, num_tokens, num_heads, d_k)
                .transpose(1, 2)
        )
        values = (
                self.value_proj(values)
                .view(batches, num_tokens, num_heads, d_v)
                .transpose(1, 2)
        )


        scores = (
                torch.einsum(
                    'bhik,bhjk->bhij',
                    queries,
                    keys,
                )
                / math.sqrt(d_k)
        )
        output_toks = (
                torch.einsum(
                        'bhis,bhsv->bhiv',
                        F.softmax(scores, dim=-1),
                        values,
                )
                .tranpose(1,2)
                .view(batches, num_tokens, num_heads * d_v)
        )



    class EncoderBlock(nn.Module):
        def __init__(self, num_heads, d_model, d_internal, d_ff):
            super().__init__()
            ### state variables
            self.d_k = d_v =  d_internal // num_heads
            self.d_internal = d_internal

            ### modules
            self.mha = MultiHeadedAttention(num_heads, d_model, self.d_k, d_v)
            self.proj = nn.Sequential(
                    [
                        nn.Linear(d_internal, d_ff),
                        nn.GELU(),
                        nn.Linear(d_ff, d_model),
                        nn.GELU(),
                    ]
            )


        def forward(self, tokens):
            d_k = self.d_k
            d_internal = self.d_internal

            tokens = F.layer_norm(tokens, d_k) 
            tokens = (
                    self.mha(tokens, tokens, tokens)
                    + tokens
            )
            tokens = F.layer_norm(tokens, d_internal)
            tokens = (
                    self.proj(tokens)
                    + tokens
            )
            return tokens

class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.encoder_blocks = nn.Sequential(
                [EncoderBlock()] * num_blocks
        )

    # I like this workflow where I do the forward function
    # first and then do __init__: basically, the important
    # part is figuring out what this module does before
    # I do the bookkeeping of making sure the object 
    # actually has the members at run-time
    def forward(self, tokens):
        return self.encoder_blocks(tokens)




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


