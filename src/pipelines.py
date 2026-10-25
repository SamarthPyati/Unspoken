"""The methods being compared. Each returns a dict with the same fields.

  single_cot   one model, one prompt, text chain-of-thought        (baseline 1)
  text_mas     planner -> critic -> refiner -> judger, text handoff (baseline 2)
  latent_mas   same four roles, latent thinking + KV-cache handoff  (proposed)
  hetero_mas   latent agents on model A, judger on model B          (stretch goal)
"""
import time
from typing import Dict, Optional

import torch

from .latent import Counter, new_cache, latent_think, decode
from .modeling import encode_chat
from .prompts import PIPELINE, build_messages, single_agent_messages


def _device(model):
    return next(model.parameters()).device


class _Timer:
    """Wall-clock + peak GPU memory, with CUDA sync so timings are honest."""

    def __init__(self, *devices):
        # hetero_mas may put its two models on different GPUs: sync all of them.
        self.cuda = sorted({d for d in devices if d.type == "cuda"}, key=str)
        self.mps = any(d.type == "mps" for d in devices)

    def _sync(self):
        for d in self.cuda:
            torch.cuda.synchronize(d)
        if self.mps:
            torch.mps.synchronize()

    def __enter__(self):
        self._sync()
        for d in self.cuda:
            torch.cuda.reset_peak_memory_stats(d)
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._sync()
        self.seconds = time.perf_counter() - self.t0
        # Peak memory is only measurable on CUDA (summed over GPUs used); 0 elsewhere.
        self.peak_mb = sum(torch.cuda.max_memory_allocated(d) for d in self.cuda) / 2**20


def _hooks(cfg, role):
    """Optional observers for the live demo: cfg.callback(role, event, value), with event
    "token" (value = token id) or "latent_step". Absent in experiments, and never able to
    change what is generated, so measured runs are unaffected."""
    cb = getattr(cfg, "callback", None)
    if cb is None:
        return None, None
    return (lambda tok: cb(role, "token", tok)), (lambda: cb(role, "latent_step", None))


def _result(output: str, counter: Counter, timer: _Timer, trace: Dict) -> Dict:
    return {
        "output": output,
        "output_tokens": counter.output_tokens,
        "per_agent_tokens": counter.per_agent_tokens,
        "prompt_tokens": counter.prompt_tokens,
        "forward_passes": counter.forward_passes,
        "latent_steps": counter.latent_steps,
        "latency_s": round(timer.seconds, 4),
        "peak_mem_mb": round(timer.peak_mb, 1),
        "trace": trace,
    }


def single_cot(model, tok, question, cfg) -> Dict:
    dev, c = _device(model), Counter()
    with _Timer(dev) as t:
        ids = encode_chat(tok, single_agent_messages(question), dev)
        out = decode(model, tok, ids, new_cache(), c, cfg.max_new_tokens,
                     cfg.temperature, cfg.generator, on_token=_hooks(cfg, "judger")[0])
    return _result(out, c, t, {})


def text_mas(model, tok, question, cfg) -> Dict:
    """Every agent decodes text; the transcript so far is pasted into the next prompt.
    Each agent starts from a fresh cache -- text is the only channel between them."""
    dev, c = _device(model), Counter()
    transcript, trace = "", {}
    with _Timer(dev) as t:
        for role in PIPELINE:
            ids = encode_chat(tok, build_messages(role, question, context=transcript), dev)
            budget = cfg.max_new_tokens if role == "judger" else cfg.agent_max_tokens
            text = decode(model, tok, ids, new_cache(), c, budget, cfg.temperature, cfg.generator,
                          on_token=_hooks(cfg, role)[0])
            trace[role] = text
            if role != "judger":
                transcript += f"[{role.title()}]\n{text}\n\n"
    return _result(text, c, t, trace)


def latent_mas(model, tok, question, cfg, w_a, target_norm) -> Dict:
    """Non-final agents think latently; ONE shared cache carries their work forward.
    Only the judger produces text.

    m = 0 has two meanings (HANDOFF issue 6), both available as ablation rows:
      cfg.m0_cache == "keep"  earlier agents' prompts stay in the cache (our default)
      cfg.m0_cache == "drop"  the judger starts from an empty cache, as in the
                              LatentMAS reference (it drops the cache when m == 0)
    """
    dev, c = _device(model), Counter()
    cache = new_cache()
    drop = cfg.latent_steps == 0 and getattr(cfg, "m0_cache", "keep") == "drop"
    with _Timer(dev) as t:
        for role in ([] if drop else PIPELINE[:-1]):
            ids = encode_chat(tok, build_messages(role, question, latent=True), dev)
            latent_think(model, ids, cache, cfg.latent_steps, w_a, target_norm, c,
                         on_step=_hooks(cfg, role)[1])
        # Read the cache length BEFORE the judger runs: this is the size of the
        # latent message handed to the judger (earlier agents' prompts + latent
        # steps). After decode it would also include the judger's own prompt
        # and generated tokens (HANDOFF issue 2).
        cache_len_before_judger = cache.get_seq_length()
        ids = encode_chat(tok, build_messages("judger", question, latent=True), dev)
        out = decode(model, tok, ids, cache, c, cfg.max_new_tokens, cfg.temperature, cfg.generator,
                     on_token=_hooks(cfg, "judger")[0])
    return _result(out, c, t, {"cache_len_before_judger": cache_len_before_judger})


def hetero_mas(model_a, model_b, tok, question, cfg, w_a, norm_a, cross_map, norm_b) -> Dict:
    """Stretch goal. KV caches cannot move between different models, so instead the
    hidden states A produced while thinking are mapped into B's input-embedding space
    by `cross_map` (see adapter.py) and fed to B as a soft prefix.
    Requires both models to share a tokenizer (e.g. two Qwen2.5 sizes)."""
    dev_a, dev_b, c = _device(model_a), _device(model_b), Counter()
    cache_a, thoughts = new_cache(), []
    with _Timer(dev_a, dev_b) as t:
        for role in PIPELINE[:-1]:
            ids = encode_chat(tok, build_messages(role, question, latent=True), dev_a)
            thoughts += latent_think(model_a, ids, cache_a, cfg.latent_steps,
                                     w_a, norm_a, c, collect=True)
        h = torch.cat(thoughts, dim=0).float().to(dev_b)               # [k, d_a]
        mapped = h @ cross_map                                         # [k, d_b]
        mapped = mapped * (norm_b / mapped.norm(dim=-1, keepdim=True).clamp_min(1e-6))
        prefix = mapped.to(next(model_b.parameters()).dtype).unsqueeze(0)  # [1, k, d_b]
        ids = encode_chat(tok, build_messages("judger", question, latent=True), dev_b)
        out = decode(model_b, tok, ids, new_cache(), c, cfg.max_new_tokens,
                     cfg.temperature, cfg.generator, inputs_embeds_prefix=prefix)
    return _result(out, c, t, {"n_thought_vectors": len(thoughts)})
