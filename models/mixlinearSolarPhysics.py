import torch
from torch import nn

from models.mixlinear import Model as MixLinearBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(MixLinearBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'mixlinearSolarPhysics',
            configs.enc_in * 32,
        )
        self._init_history_context(configs)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, future_zenith=None):
        base_out = super(Model, self).forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)
        if future_zenith is None:
            return base_out
        context = self.history_context(x_enc.reshape(x_enc.shape[0], -1))
        return self._apply_physics(context, base_out, future_zenith)
