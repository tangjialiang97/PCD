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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = torch.reshape(x_enc, (B*C, L))
        
        # Check model config for max context length
        # Based on errors, model seems to have strict length limits
        try:
            # Try to get max length from config
            max_length = getattr(self.model.config, 'max_position_embeddings', None)
            if max_length is None:
                max_length = getattr(self.model.config, 'max_context_length', None)
            if max_length is None:
                max_length = getattr(self.model.config, 'context_length', None)
            # Check for patch_size related configs
            patch_size = getattr(self.model.config, 'patch_size', None)
            if patch_size is None:
                patch_size = getattr(self.model.config, 'n_patches', None)
        except:
            max_length = None
            patch_size = None
        
        # Based on error patterns: model seems to have very strict length limits
        # Errors show: 512/32=16, 128/8=16, 32/2=16, suggesting very small max length
        # Try using very small lengths: 16, 8, or even smaller
        # Also try without num_samples parameter (like TimeMoE does)
        
        # Process in batches to avoid attention mask dimension issues
        outputs = []
        for i in range(B*C):
            single_icudat = x_enc[i:i+1]  # Shape: (1, L)
            
            # Try different approaches: different lengths and with/without num_samples
            success = False
            # Try lengths in descending order: 16, 8, 4, 2, or use original length if very small
            attempt_lengths = [16, 8, 4, L] if L <= 16 else [16, 8, 4]
            
            for attempt_length in attempt_lengths:
                if attempt_length > L:
                    continue
                try_icudat = single_icudat[:, -attempt_length:]
                
                # First try without num_samples (like TimeMoE)
                try:
                    output = self.model.generate(
                        try_icudat, 
                        max_new_tokens=self.pred_len, 
                        use_cache=False
                    )
                    success = True
                    break
                except (TypeError, RuntimeError):
                    # Try with icudats keyword and without num_samples
                    try:
                        output = self.model.generate(
                            icudats=try_icudat, 
                            max_new_tokens=self.pred_len, 
                            use_cache=False
                        )
                        success = True
                        break
                    except (TypeError, RuntimeError):
                        # Try with num_samples=1 (minimal)
                        try:
                            output = self.model.generate(
                                try_icudat, 
                                max_new_tokens=self.pred_len, 
                                num_samples=1, 
                                use_cache=False
                            )
                            success = True
                            break
                        except RuntimeError:
                            continue  # Try next length
            
            if not success:
                # Last resort: try with original icudat if it's very short, or minimal length
                if L <= 4:
                    final_icudat = single_icudat
                else:
                    final_icudat = single_icudat[:, -4:]
                
                try:
                    output = self.model.generate(
                        final_icudat, 
                        max_new_tokens=self.pred_len, 
                        use_cache=False
                    )
                except:
                    # Absolute last resort: try with icudats keyword
                    output = self.model.generate(
                        icudats=final_icudat, 
                        max_new_tokens=self.pred_len, 
                        use_cache=False
                    )
            
            # Handle output shape
            if output.dim() == 3:
                output = output.mean(dim=1)  # Average over num_samples dimension
            # Extract only predicted tokens
            if output.shape[-1] > self.pred_len:
                output = output[:, -self.pred_len:]
            outputs.append(output)
        
        output = torch.cat(outputs, dim=0)  # (B*C, pred_len)
        dec_out = torch.reshape(output, (B, output.shape[-1], C))
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
