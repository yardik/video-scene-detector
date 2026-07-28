import sys
import torch
import torch.nn.functional as F

_orig_sdpa = F.scaled_dot_product_attention
_sage_active = False


def sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    try:
        if query.dtype in (torch.float16, torch.bfloat16) and query.is_cuda:
            from sageattention import sageattn, sageattn_varlen
            is_causal_bool = is_causal or (attn_mask is not None)

            # Handle 3D variable-length unpadded tensors (total_tokens, num_heads, head_dim)
            if query.dim() == 3 and "cu_seqlens_q" in kwargs:
                return sageattn_varlen(
                    query, key, value,
                    cu_seqlens_q=kwargs["cu_seqlens_q"],
                    cu_seqlens_k=kwargs["cu_seqlens_k"],
                    max_seqlen_q=kwargs.get("max_seqlen_q", query.size(0)),
                    max_seqlen_k=kwargs.get("max_seqlen_k", key.size(0)),
                    is_causal=is_causal_bool,
                )

            # Handle 4D batched tensors [batch, num_heads, seq_len, head_dim]
            if query.dim() == 4:
                q_nhd = query.transpose(1, 2).contiguous()
                k_nhd = key.transpose(1, 2).contiguous()
                v_nhd = value.transpose(1, 2).contiguous()
                out_nhd = sageattn(q_nhd, k_nhd, v_nhd, tensor_layout="NHD", is_causal=is_causal_bool)
                return out_nhd.transpose(1, 2)
    except Exception:
        pass
    return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs)


def enable_sage_attention() -> bool:
    global _sage_active
    try:
        import sageattention
        F.scaled_dot_product_attention = sage_sdpa_wrapper
        _sage_active = True
        print("⚡ SageAttention 2 INT8 Kernel active (varlen API & NHD layout enabled)!", file=sys.stderr)
        return True
    except Exception as e:
        print(f"⚠️ Could not enable SageAttention 2: {e}", file=sys.stderr)
        return False


def disable_sage_attention():
    global _sage_active
    F.scaled_dot_product_attention = _orig_sdpa
    _sage_active = False
    print("⚡ Reverted to Native PyTorch SDPA Tensor Core kernel.", file=sys.stderr)


def is_sage_attention_active() -> bool:
    return _sage_active
