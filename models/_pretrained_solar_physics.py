import torch
from torch import nn

from models._solar_physics_utils import SolarPhysicsMixin


class FrozenPretrainedSolarPhysicsModel(nn.Module, SolarPhysicsMixin):
    def __init__(self, configs, model_name, backbone, default_hidden_dim):
        super().__init__()
        self.backbone = backbone
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self._freeze_backbone()
        self._init_solar_physics(configs, model_name, default_hidden_dim)
        self._init_history_context(configs)

    def _freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _forecast_backbone(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        with torch.no_grad():
            return self.backbone.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, future_zenith=None):
        self._require_future_zenith(future_zenith)
        base_out = self._forecast_backbone(x_enc, x_mark_enc, x_dec, x_mark_dec)
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        return self._apply_physics(context, base_out, future_zenith)

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
        return self.backbone(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
