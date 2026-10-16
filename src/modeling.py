"""Model loading and latent-space realignment.

Realignment follows the LatentMAS reference implementation
(github.com/Gen-Verse/LatentMAS, models.py):

    W_a = argmin_W ||W_out W - W_in||^2 + lambda ||W||^2
        = (W_out^T W_out + lambda I)^(-1) W_out^T W_in

W_out (vocab x d) is the LM head and W_in (vocab x d) the input embedding table.
A last-layer hidden state h is mapped to h @ W_a, then rescaled to the mean
norm of the real input embeddings, so the next forward pass sees an input
that looks like a token embedding instead of an out-of-distribution vector.
Three modes, for the ablation:
  "full"       W_a as above + norm rescaling   (LatentMAS with realignment on)
  "norm_only"  identity + norm rescaling       (LatentMAS default with it off)
  "raw"        hidden state fed back unchanged (no correction at all)

IMPORTANT: small Qwen2.5 models (0.5B/1.5B/3B) TIE input and output embeddings,
so W_out == W_in and W_a collapses to ~identity. For those models "full" and
"norm_only" are nearly the same; compare against "raw" instead. Check
model.config.tie_word_embeddings and report it in your write-up.
"""
import os
from typing import Optional, Tuple

import torch
import transformers
from packaging import version  # already a transformers dependency
from transformers import AutoModelForCausalLM, AutoTokenizer


def default_device() -> str:
    """cuda (Kaggle T4, the target) > mps (Apple GPU, for local debugging) > cpu.

    Final numbers come from CUDA runs only: peak memory is measured on CUDA only,
    and kernels differ across backends, so outputs are not bit-identical."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_kwarg(dtype: torch.dtype) -> dict:
    """The from_pretrained keyword for the weight dtype, for this transformers version.

    transformers renamed `torch_dtype=` to `dtype=` in 4.56 and now warns on the
    old name. We pick by version instead of try/except on purpose: older versions
    silently forward unknown kwargs to the model config, so passing `dtype=` there
    would be IGNORED and the model would load in fp32 without any error.
    """
    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def load_model(name: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, **dtype_kwarg(dtype)).to(device).eval()
    return model, tokenizer


def encode_chat(tokenizer, messages, device) -> torch.Tensor:
    """Chat-template the messages and return input_ids of shape [1, T]."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)


@torch.no_grad()
def build_realign(model, mode: str = "full", ridge: float = 1e-5,
                  cache_dir: str = "cache") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return (W_a [d x d], target_norm scalar), both fp32 on the model's device.

    The solve is done once on CPU in fp32 and saved to disk, because the
    vocab x d matrices are large (about 1.2 GB in fp32 for a 3B model) and
    a T4 does not need that extra memory spike during every run.
    """
    device = next(model.parameters()).device
    w_in = model.get_input_embeddings().weight
    d = w_in.shape[1]
    tag = getattr(model.config, "_name_or_path", "model").replace("/", "_")
    path = os.path.join(cache_dir, f"realign_{tag}_{ridge}.pt")

    if os.path.exists(path):
        saved = torch.load(path)
        w_a, target_norm = saved["w_a"], saved["target_norm"]
    else:
        w_in32 = w_in.detach().float().cpu()
        w_out32 = model.get_output_embeddings().weight.detach().float().cpu()
        gram = w_out32.T @ w_out32 + ridge * torch.eye(d)
        w_a = torch.linalg.solve(gram, w_out32.T @ w_in32)
        target_norm = w_in32.norm(dim=1).mean()
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"w_a": w_a, "target_norm": target_norm}, path)

    if mode == "raw":
        return torch.eye(d, device=device), None
    if mode == "norm_only":
        w_a = torch.eye(d)
    elif mode != "full":
        raise ValueError(f"unknown realign mode: {mode}")
    return w_a.to(device), target_norm.to(device)


def apply_realign(hidden: torch.Tensor, w_a: torch.Tensor,
                  target_norm: Optional[torch.Tensor]) -> torch.Tensor:
    """hidden [B, d] -> aligned input embedding [B, d], same dtype as hidden."""
    if target_norm is None:  # "raw" mode
        return hidden
    aligned = hidden.float() @ w_a
    aligned = aligned * (target_norm / aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    return aligned.to(hidden.dtype)
