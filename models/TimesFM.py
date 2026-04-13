import torch
import numpy as np
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timesfm
from models._hf_local import resolve_hf_file


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        repo_id = "google/timesfm-2.5-200m-pytorch"
        cache_dir = getattr(configs, 'hf_cache_dir', None)
        model_path = resolve_hf_file(repo_id, "model.safetensors", cache_dir=cache_dir)
        # Resolve the config once so subsequent loads remain fully local.
        resolve_hf_file(repo_id, "config.json", cache_dir=cache_dir)
        self.model = timesfm.TimesFM_2p5_200M_torch()
        self.model.model.load_checkpoint(
            model_path,
            torch_compile=bool(getattr(configs, 'timesfm_torch_compile', True)),
        )
        per_core_batch_size = max(1, int(getattr(configs, 'timesfm_per_core_batch_size', 128)))
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=configs.seq_len,
                max_horizon=configs.pred_len,
                normalize_inputs=False,
                per_core_batch_size=per_core_batch_size,
                use_continuous_quantile_head=bool(
                    getattr(configs, 'timesfm_use_continuous_quantile_head', False)
                ),
                force_flip_invariance=bool(
                    getattr(configs, 'timesfm_force_flip_invariance', False)
                ),
                infer_is_positive=bool(
                    getattr(configs, 'timesfm_infer_is_positive', False)
                ),
                fix_quantile_crossing=bool(
                    getattr(configs, 'timesfm_fix_quantile_crossing', False)
                ),
            )
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device
        # TimesFM expects a list of univariate numpy arrays, one per sample/channel.
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, L)
        inputs = list(x_enc.detach().cpu().numpy().astype(np.float32, copy=False))

        output, _ = self.model.forecast(
            horizon=self.pred_len,
            inputs=inputs,
        )
        output = torch.as_tensor(output, device=device, dtype=x_enc.dtype)

        dec_out = output.reshape(B, C, self.pred_len).permute(0, 2, 1).contiguous()
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
