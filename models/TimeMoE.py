import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
from transformers import AutoModelForCausalLM, StoppingCriteriaList, MaxLengthCriteria
from models._hf_local import resolve_hf_snapshot

class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        repo_id = 'Maple728/TimeMoE-50M'
        cache_dir = getattr(configs, 'hf_cache_dir', None)
        local_snapshot = resolve_hf_snapshot(repo_id, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            local_snapshot,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_cache = bool(getattr(configs, 'time_moe_use_cache', True))

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        device = x_enc.device
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, L)

        generation_config = self.model.generation_config
        generation_config.do_sample = False
        generation_config.num_beams = 1
        stopping_criteria = StoppingCriteriaList(
            [MaxLengthCriteria(max_length=L + self.pred_len)]
        )

        try:
            output = self.model._greedy_search(
                x_enc,
                pad_token_id=generation_config.pad_token_id,
                eos_token_id=generation_config.eos_token_id,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=False,
                use_cache=self.use_cache,
            )
        except (RuntimeError, AttributeError) as e:
            if "dimensions" in str(e) or "dim" in str(e).lower() or "got 3 and 2" in str(e) or "has no attribute" in str(e):
                output = self.model.generate(
                    x_enc,
                    max_new_tokens=self.pred_len,
                    use_cache=self.use_cache,
                    do_sample=False,
                    num_beams=1,
                    return_dict_in_generate=False
                )
            else:
                raise e

        output = output.to(device)
        dec_out = output.reshape(B, C, output.shape[-1]).permute(0, 2, 1).contiguous()
        dec_out = dec_out[:, -self.pred_len:, :]
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
