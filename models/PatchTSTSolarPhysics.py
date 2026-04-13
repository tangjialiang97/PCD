import torch
from torch import nn

from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.solar_physics import SolarPhysicsStack
from models.PatchTST import Transpose


class FlattenChannelHead(nn.Module):
    def __init__(self, nf, out_dim, head_dropout=0):
        super(FlattenChannelHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(nf, out_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PerVariableForecastHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super(PerVariableForecastHead, self).__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        if self.enc_in != 3:
            raise ValueError('PatchTSTSolarPhysics expects exactly 3 target channels: [ghi, dni, dhi].')
        patch_len = int(getattr(configs, 'patch_len', patch_len))
        self.physics_hidden_dim = int(getattr(configs, 'physics_hidden_dim', configs.d_model))
        self.physics_layers = int(getattr(configs, 'physics_layers', 2))
        self.night_cos_threshold = float(getattr(configs, 'night_cos_threshold', 0.0))
        self.physics_eps = float(getattr(configs, 'physics_eps', 1e-4))

        padding = stride
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2),
            ),
        )

        patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * patch_num
        self.base_head = PerVariableForecastHead(
            configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout
        )
        self.context_head = FlattenChannelHead(
            configs.enc_in * self.head_nf,
            configs.pred_len * self.physics_hidden_dim,
            head_dropout=configs.dropout,
        )
        self.base_state_embed = nn.Linear(3, self.physics_hidden_dim)
        self.physics_stack = SolarPhysicsStack(
            input_dim=self.physics_hidden_dim,
            hidden_dim=self.physics_hidden_dim,
            num_layers=self.physics_layers,
            dropout=configs.dropout,
            kernel_size=int(getattr(configs, 'physics_kernel_size', 3)),
            night_cos_threshold=self.night_cos_threshold,
            eps=self.physics_eps,
            use_gated_feedback=bool(getattr(configs, 'physics_gated_feedback', False)),
            use_weak_upper_bound=bool(getattr(configs, 'physics_use_weak_upper_bound', False)),
            solar_constant=float(getattr(configs, 'solar_constant', 1367.0)),
            upper_bound_alpha_min=float(getattr(configs, 'physics_upper_bound_alpha_min', 0.8)),
            upper_bound_alpha_max=float(getattr(configs, 'physics_upper_bound_alpha_max', 1.2)),
        )

    def _compute_cos_zenith(self, future_zenith):
        return torch.cos(torch.deg2rad(future_zenith)).clamp(min=0.0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        if future_zenith is None:
            raise ValueError('PatchTSTSolarPhysics requires future_zenith for forecast.')

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out, _attns = self.encoder(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        base_out = self.base_head(enc_out).permute(0, 2, 1)
        base_out = base_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        base_out = base_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        context = self.context_head(enc_out.reshape(enc_out.shape[0], -1))
        context = context.reshape(enc_out.shape[0], self.pred_len, self.physics_hidden_dim)
        context = context + self.base_state_embed(base_out)

        cos_zenith = self._compute_cos_zenith(future_zenith)
        dec_out, _hidden = self.physics_stack(context, cos_zenith)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(
                x_enc,
                x_mark_enc,
                x_dec,
                x_mark_dec,
                future_zenith=future_zenith,
            )
            return dec_out[:, -self.pred_len:, :]
        return None
