import torch

from models.EMAformer import Model as EMAformerBackbone
from models._solar_physics_utils import SolarPhysicsMixin


class Model(EMAformerBackbone, SolarPhysicsMixin):
    def __init__(self, configs):
        super(Model, self).__init__(configs)
        self._init_solar_physics(
            configs,
            'EMAformerSolarPhysics',
            getattr(configs, 'd_model', configs.enc_in * 32),
        )
        self._init_flat_context(configs.d_model)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, future_zenith=None):
        cycle_index = torch.zeros(x_enc.shape[0], dtype=torch.long, device=x_enc.device)

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc = x_enc / stdev

        B, L, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        channel_emb = self.channel_embedding.expand(B, N, -1)
        phase_emb = self.phase_embedding(cycle_index.view(-1, 1).expand(B, N))
        joint_emb = self.joint_embedding(cycle_index).reshape(B, self.enc_in, self.d_model)
        enc_out = enc_out[:, :N, :] + channel_emb + phase_emb + joint_emb
        enc_origin = enc_out

        enc_out, _attns = self.encoder(enc_out, attn_mask=None)

        base_out = self.projector(enc_out + enc_origin).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            base_out = base_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            base_out = base_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)

        if future_zenith is None:
            return base_out[:, -self.pred_len:, :]

        context = self.context_head(enc_out.mean(dim=1))
        return self._apply_physics(context, base_out[:, -self.pred_len:, :], future_zenith)
