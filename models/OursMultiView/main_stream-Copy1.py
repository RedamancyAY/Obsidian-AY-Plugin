# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from ay2.tools.torch_model import freeze_modules
from einops import rearrange
from transformers import AutoFeatureExtractor, WavLMModel

# %%
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): (B, C, L) from the 1D stream
            y (torch.Tensor): (B, C, L) from the 2D stream

        Returns:
            torch.Tensor: the fused feature
        """
        x = rearrange(x, "b c l -> b l c")
        y = rearrange(y, "b c l -> b l c")

        out = x + y
        attn_output, attn_output_weights = self.mha(x, y, out)
        out = out + attn_output

        out = self.layer_norm1(out)
        out = out + self.mlp(out)
        out = self.layer_norm2(out)

        out = rearrange(out, "b l c -> b c l")

        return out


# %%
# m = TransformerLayer(embed_dim=64)
# x = torch.randn(2, 64, 12000)
# out = m(x, x)
# print(out.shape)

# %%
class WavLMFeatureProjection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, 768)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(norm_hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# %%
class MainStream(nn.Module):
    def __init__(
        self, embed_dims=[64, 128, 256, 512], verbose=0, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embed_dims = embed_dims
        self.verbose = verbose
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(embed_dim=x) for x in embed_dims]
        )
        self.conv1Ds = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(_c, _c, 1),
                    nn.BatchNorm1d(_c),
                    nn.ReLU(_c),
                    nn.Conv1d(_c, _c, 1),
                )
                for _c in embed_dims
            ]
        )

        self.projections = nn.ModuleList(
            [
                nn.Conv1d(_c, embed_dims[i + 1], 3, stride=2, padding=1)
                for i, _c in enumerate(embed_dims[0:-1])
            ]
        )
        self.final_bn = nn.BatchNorm1d(768)
        self.final_relu = nn.ReLU()

        self.downsample_layers1, self.downsample_layers2 = [
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv1d(d, d, k, stride=s, padding=p),
                        nn.BatchNorm1d(d),
                        nn.Dropout1d(0.1),
                    )
                    for d, k, s, p in zip(
                        embed_dims, [3, 3, 3, 3], [2, 2, 2, 2], [1, 1, 1, 1]
                    )
                ]
            )
            for _ in range(2)
        ]

        self.final_downsample = nn.Conv1d(
            embed_dims[-1], embed_dims[-1], 10, stride=5, padding=3
        )

        self.featureProjection = WavLMFeatureProjection(256)
        self.transformer = WavLMModel.from_pretrained(
            "/usr/local/ay_data/0-model_weights/microsoft_wavlm-base"
        ).encoder
        # freeze_modules(self.transformer)

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10.0
        code = torch.div(code, code_norm)
        return code

    def compute_latent_feature(self, out):
        # final prediction
        out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)

        out = self.feature_norm(out)
        if self.verbose:
            print("Main Stream => Latent Feature: output shape", out.shape)

        return out

    def compute_stage(
        self, feat1D: torch.Tensor, feat2D: torch.Tensor, stage_idx: int
    ) -> torch.Tensor:
        """_summary_

        Args:
            feat1D (torch.Tensor): _description_
            feat2D (torch.Tensor): _description_
            stage_idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.verbose:
            print(
                f"Main Stream => stage {stage_idx+1} input shape",
                feat1D.shape,
                feat2D.shape,
            )

        h, w = feat2D.shape[-2:]
        L = feat1D.shape[-1]
        scale_factor = L / (h * w)

        feat2D = rearrange(feat2D, "b c h w -> b c (w h)")
        feat2D = F.upsample_nearest(feat2D, scale_factor=scale_factor + 0.0001)
        feat2D = self.conv1Ds[stage_idx](feat2D)

        feat2D = self.downsample_layers2[stage_idx](feat2D)
        feat1D = self.downsample_layers1[stage_idx](feat1D)
        out = feat1D + feat2D

        if stage_idx == 0:
            self.previous_out = out
        else:
            self.previous_out = self.projections[stage_idx - 1](self.previous_out) + out
            if stage_idx + 1 == len(self.embed_dims):
                out = self.final_downsample(self.previous_out)
                out = rearrange(out, "b c l -> b l c")
                out = self.featureProjection(out)
                encoder_outputs = self.transformer(
                    out,
                    attention_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=False,
                )
                out = encoder_outputs[0]
                self.previous_out = rearrange(out, "b l c -> b c l")

        if self.verbose:
            print(
                f"Main Stream => stage {stage_idx+1} output shape",
                self.previous_out.shape,
            )

        return self.previous_out

    def compute_final_stage(
        self, feat1D: torch.Tensor, feat2D: torch.Tensor, stage_idx: int = 3
    ) -> torch.Tensor:
        """_summary_

        Args:
            feat1D (torch.Tensor): _description_
            feat2D (torch.Tensor): _description_
            stage_idx (int): _description_

        Returns:
            torch.Tensor: _description_
        """
        if self.verbose:
            print(
                f"Main Stream => stage {stage_idx+1} input shape",
                feat1D.shape,
                feat2D.shape,
            )

        h, w = feat2D.shape[-2:]
        L = feat1D.shape[-1]
        scale_factor = L / (h * w)

        feat2D = rearrange(feat2D, "b c h w -> b c (w h)")
        feat2D = F.upsample_nearest(feat2D, scale_factor=scale_factor + 0.0001)
        feat2D = self.conv1Ds[stage_idx](feat2D)

        feat2D = self.downsample_layers2[stage_idx](feat2D)
        feat1D = self.downsample_layers1[stage_idx](feat1D)
        out = feat1D + feat2D

        out = self.final_downsample(out)
        out = rearrange(out, "b c l -> b l c")
        out = self.featureProjection(out)
        # encoder_outputs = self.transformer(
        #     out,
        #     attention_mask=None,
        #     output_attentions=False,
        #     output_hidden_states=False,
        #     return_dict=False,
        # )
        # out = encoder_outputs[0]
        out = rearrange(out, "b l c -> b c l")

        if self.verbose:
            print(
                f"Main Stream => stage {stage_idx+1} output shape",
                self.previous_out.shape,
            )

        return out

    def apply_transformer(self, feat1D):

        out =feat1D
        out = rearrange(out, "b c l -> b l c")
        out = self.featureProjection(out)
        encoder_outputs = self.transformer(
            out,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        out = encoder_outputs[0]
        out = rearrange(out, "b l c -> b c l")
        return out
# %%
# model = MainStream()
# x = torch.randn(2, 64, 12000)
# y = torch.randn(2, 64,65, 65)
# model.compute_stage(x, y, 1)
