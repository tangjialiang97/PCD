import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModelForCausalLM

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained('thuml/sundial-base-128m', trust_remote_code=True)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def _generate(self, series_batch):
        input_token_len = int(getattr(self.model.config, 'input_token_len', 16))
        token_len = (series_batch.shape[-1] + input_token_len - 1) // input_token_len
        attention_mask = torch.ones(
            series_batch.shape[0],
            token_len,
            dtype=torch.long,
            device=series_batch.device,
        )
        output = self.model.generate(
            inputs=series_batch,
            attention_mask=attention_mask,
            max_new_tokens=self.pred_len,
            num_samples=1,
        )
        if output.dim() == 3:
            output = output.mean(dim=1)
        if output.dim() != 2:
            raise RuntimeError(
                "Unexpected Sundial output shape: {}".format(tuple(output.shape))
            )
        if output.shape[-1] > self.pred_len:
            output = output[:, -self.pred_len:]
        if output.shape[-1] != self.pred_len:
            raise RuntimeError(
                "Sundial returned {} forecast steps, expected {}".format(
                    output.shape[-1], self.pred_len
                )
            )
        return output

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        B, L, C = x_enc.shape
        outputs = []
        for i in range(C):
            series_batch = x_enc[:, :, i]
            output = self._generate(series_batch)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
