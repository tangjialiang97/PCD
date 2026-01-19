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
        self.model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', trust_remote_code=True)
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
        
        # Process each sequence individually to avoid dimension mismatch issues
        outputs = []
        for i in range(B*C):
            single_input = x_enc[i:i+1]  # Shape: (1, L) - ensure 2D
            # Ensure input is contiguous and has correct dtype
            single_input = single_input.contiguous()
            
            try:
                # Force use of _greedy_search by setting generation config
                # TimeMoE has custom _greedy_search that handles 3D tensors correctly
                generation_config = self.model.generation_config
                generation_config.do_sample = False
                generation_config.num_beams = 1
                
                # Use max_new_tokens instead of max_length
                from transformers import StoppingCriteriaList, MaxLengthCriteria
                stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=single_input.shape[1] + self.pred_len)])
                
                # Directly call _greedy_search to avoid transformers' _sample method
                # Pass use_cache through model_kwargs
                output = self.model._greedy_search(
                    single_input,
                    max_length=single_input.shape[1] + self.pred_len,
                    pad_token_id=generation_config.pad_token_id,
                    eos_token_id=generation_config.eos_token_id,
                    stopping_criteria=stopping_criteria,
                    return_dict_in_generate=False,
                    use_cache=False  # Pass through model_kwargs
                )
                outputs.append(output)
            except (RuntimeError, AttributeError) as e:
                # Fallback to standard generate if _greedy_search fails
                if "dimensions" in str(e) or "dim" in str(e).lower() or "got 3 and 2" in str(e) or "has no attribute" in str(e):
                    try:
                        # Try with explicit greedy search parameters
                        output = self.model.generate(
                            single_input, 
                            max_new_tokens=self.pred_len, 
                            use_cache=False,
                            do_sample=False,
                            num_beams=1,
                            return_dict_in_generate=False
                        )
                        outputs.append(output)
                    except RuntimeError:
                        # Last resort: try with different parameters
                        output = self.model.generate(
                            single_input, 
                            max_length=single_input.shape[1] + self.pred_len,
                            use_cache=False,
                            do_sample=False,
                            num_beams=1
                        )
                        outputs.append(output)
                else:
                    raise e
        
        # Concatenate all outputs
        output = torch.cat(outputs, dim=0)  # Shape: (B*C, seq_len + pred_len)
        dec_out = torch.reshape(output, (B, output.shape[-1], C))
        dec_out = dec_out[:,-self.pred_len:, :]
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
