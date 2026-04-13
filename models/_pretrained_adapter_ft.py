import torch
from torch import nn


class ForecastBottleneckAdapter(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, dropout=0.1):
        super().__init__()
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up_proj = nn.Linear(bottleneck_dim, hidden_dim)

        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        x = self.pre_norm(x)
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        x = self.dropout(x)
        return x


class FrozenPretrainedAdapterFTModel(nn.Module):
    def __init__(self, configs, model_name, backbone, default_hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self._freeze_backbone()
        self._init_adapter(configs, model_name, default_hidden_dim)

    def _freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _init_adapter(self, configs, model_name, default_hidden_dim):
        self.ft_model_name = model_name
        self.adapter_hidden_dim = int(
            getattr(configs, 'ft_adapter_hidden_dim', default_hidden_dim)
        )
        default_bottleneck_dim = max(8, self.adapter_hidden_dim // 4)
        self.adapter_bottleneck_dim = int(
            getattr(configs, 'ft_adapter_bottleneck_dim', default_bottleneck_dim)
        )
        self.adapter_dropout = float(
            getattr(configs, 'ft_adapter_dropout', getattr(configs, 'dropout', 0.1))
        )

        self.history_context = nn.Linear(
            configs.seq_len * configs.enc_in,
            self.pred_len * self.adapter_hidden_dim,
        )
        self.base_state_embed = nn.Linear(configs.enc_in, self.adapter_hidden_dim)
        self.adapter = ForecastBottleneckAdapter(
            self.adapter_hidden_dim,
            self.adapter_bottleneck_dim,
            dropout=self.adapter_dropout,
        )
        self.output_proj = nn.Linear(self.adapter_hidden_dim, configs.enc_in)
        nn.init.zeros_(self.output_proj.bias)

    def _forecast_backbone(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            return self.backbone.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        base_out = self._forecast_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        context = context.reshape(x_enc.shape[0], self.pred_len, self.adapter_hidden_dim)
        hidden = context + self.base_state_embed(base_out)
        residual = self.output_proj(self.adapter(hidden))
        return base_out + residual

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
