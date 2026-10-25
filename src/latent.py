"""The three primitive operations every pipeline is built from.

  prefill(...)       feed real prompt tokens, extending the KV cache
  latent_think(...)  m steps of feeding the (realigned) last hidden state back
                     in as the next input embedding -- no token is sampled
  decode(...)        ordinary token-by-token generation, continuing from a cache

All three write into a shared `Counter`, so every method is measured the same
way: generated tokens, forward passes, and latent steps.

We use a hand-written decode loop instead of model.generate() on purpose:
it makes cache handling explicit, counts every forward pass exactly, and
behaves the same across transformers versions. Batch size is 1 throughout,
which removes padding and position-id bugs (the most common way latent
experiments silently break). Batch later, once results are trusted.
"""
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set

import torch
from transformers import DynamicCache

from .modeling import apply_realign


@dataclass
class Counter:
    output_tokens: int = 0     # tokens actually decoded into text
    forward_passes: int = 0    # every model() call, prefill included
    latent_steps: int = 0      # steps spent thinking in hidden space
    prompt_tokens: int = 0     # tokens prefilled from prompts
    per_agent_tokens: List[int] = field(default_factory=list)


def new_cache() -> DynamicCache:
    return DynamicCache()


def _forward(model, cache, counter, input_ids=None, inputs_embeds=None):
    new_len = (input_ids if input_ids is not None else inputs_embeds).shape[1]
    past_len = cache.get_seq_length()
    device = next(model.parameters()).device
    attention_mask = torch.ones((1, past_len + new_len), dtype=torch.long, device=device)
    out = model(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=cache,
        use_cache=True,
        output_hidden_states=True,
        return_dict=True,
    )
    counter.forward_passes += 1
    return out


@torch.no_grad()
def prefill(model, input_ids, cache, counter):
    out = _forward(model, cache, counter, input_ids=input_ids)
    counter.prompt_tokens += input_ids.shape[1]
    return out


@torch.no_grad()
def latent_think(model, input_ids, cache, steps, w_a, target_norm, counter,
                 collect: bool = False, on_step: Optional[Callable[[], None]] = None):
    """Prefill the agent's prompt, then think for `steps` latent steps.

    Returns the list of last-layer hidden states produced (only if collect=True;
    the cross-model adapter needs them). The cache is extended in place by
    len(prompt) + steps positions -- that cache IS the message to the next agent.
    on_step (optional) is called after each latent step; it only observes, for the live demo.
    """
    out = prefill(model, input_ids, cache, counter)
    last_hidden = out.hidden_states[-1][:, -1, :]          # [1, d]
    collected = []
    for _ in range(steps):
        if collect:
            collected.append(last_hidden.detach())
        embed = apply_realign(last_hidden, w_a, target_norm).unsqueeze(1)  # [1, 1, d]
        out = _forward(model, cache, counter, inputs_embeds=embed)
        last_hidden = out.hidden_states[-1][:, -1, :]
        counter.latent_steps += 1
        if on_step is not None:
            on_step()
    if collect:
        collected.append(last_hidden.detach())
    return collected


def stop_token_ids(model, tokenizer) -> Set[int]:
    ids = set()
    eos = model.generation_config.eos_token_id
    if isinstance(eos, int):
        ids.add(eos)
    elif eos:
        ids.update(eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")  # Qwen chat end token
    if isinstance(im_end, int) and im_end != tokenizer.unk_token_id:
        ids.add(im_end)
    return ids


@torch.no_grad()
def decode(model, tokenizer, input_ids, cache, counter, max_new_tokens=512,
           temperature: float = 0.0, generator: Optional[torch.Generator] = None,
           inputs_embeds_prefix: Optional[torch.Tensor] = None,
           on_token: Optional[Callable[[int], None]] = None) -> str:
    """Prefill `input_ids` on top of `cache`, then generate text.

    temperature=0 gives greedy (deterministic) decoding -- the default for
    all main experiments. inputs_embeds_prefix lets a different model's mapped
    latent thoughts be fed in first (used only by the hetero pipeline).
    on_token (optional) is called with each generated token id; it only observes
    (the live demo streams text with it) and never changes what is generated.
    """
    if inputs_embeds_prefix is not None:
        _forward(model, cache, counter, inputs_embeds=inputs_embeds_prefix)
    out = prefill(model, input_ids, cache, counter)
    stops = stop_token_ids(model, tokenizer)
    generated: List[int] = []
    logits = out.logits[:, -1, :]
    for _ in range(max_new_tokens):
        if temperature > 0:
            probs = torch.softmax(logits.float() / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1, generator=generator)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)
        token = int(next_id.item())
        if token in stops:
            break
        generated.append(token)
        if on_token is not None:
            on_token(token)
        out = _forward(model, cache, counter, input_ids=next_id)
        logits = out.logits[:, -1, :]
    counter.output_tokens += len(generated)
    counter.per_agent_tokens.append(len(generated))
    return tokenizer.decode(generated, skip_special_tokens=True)
