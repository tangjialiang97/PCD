import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from tirex import load_model, ForecastModel
from models._hf_local import resolve_hf_download_kwargs


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.tirex_device = self._resolve_tirex_device(configs)
        self.tirex_batch_size = self._resolve_tirex_batch_size(configs)
        hf_kwargs = resolve_hf_download_kwargs("NX-AI/TiRex", "model.ckpt")
        self.model = load_model(
            "NX-AI/TiRex",
            device=self.tirex_device,
            compile=bool(getattr(configs, 'tirex_compile', False)),
            hf_kwargs=hf_kwargs,
        )
        self.model.eval()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def _resolve_tirex_device(self, configs):
        if getattr(configs, 'use_gpu', False) and getattr(configs, 'gpu_type', 'cuda') == 'cuda' and torch.cuda.is_available():
            return 'cuda:{}'.format(getattr(configs, 'gpu', 0))
        return 'cpu'

    def _resolve_tirex_batch_size(self, configs):
        configured_batch_size = int(getattr(configs, 'tirex_batch_size', 0))
        if configured_batch_size > 0:
            return configured_batch_size
        runtime_batch_size = int(getattr(configs, 'batch_size', 1))
        return max(512, runtime_batch_size * int(configs.enc_in))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, L)
        _quantiles, output = self.model.forecast(
            x_enc,
            prediction_length=self.pred_len,
            batch_size=self.tirex_batch_size,
            output_device=str(device),
        )
        output = output.to(device=device, dtype=x_enc.dtype)
        dec_out = output.reshape(B, C, self.pred_len).permute(0, 2, 1).contiguous()
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
