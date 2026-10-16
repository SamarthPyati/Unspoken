"""Mechanics tests on a tiny RANDOM Qwen2 model -- no download, runs on CPU in seconds.

Run this before any real experiment:  pytest tests/test_mechanics.py -v
If the cache-equivalence test fails, every latent result would be invalid.
"""
import pytest

try:  # skip cleanly if torch is missing OR broken (e.g. missing CUDA libs)
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"torch/transformers unavailable: {exc}", allow_module_level=True)

from src.latent import Counter, new_cache, latent_think, decode, prefill  # noqa: E402
from src.modeling import build_realign, apply_realign  # noqa: E402


class FakeTokenizer:
    eos_token_id = 0
    unk_token_id = None

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=500, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=512, tie_word_embeddings=False)
    model = Qwen2ForCausalLM(cfg).eval()
    model.generation_config.eos_token_id = 0
    return model, FakeTokenizer()


def ids(*xs):
    return torch.tensor([list(xs)])


def test_realign_shapes_and_norm(tiny, tmp_path):
    model, _ = tiny
    w_a, norm = build_realign(model, mode="full", cache_dir=str(tmp_path))
    assert w_a.shape == (64, 64)
    h = torch.randn(3, 64)
    out = apply_realign(h, w_a, norm)
    assert torch.allclose(out.norm(dim=-1), norm.expand(3), atol=1e-4)
    w_id, _ = build_realign(model, mode="norm_only", cache_dir=str(tmp_path))
    assert torch.equal(w_id, torch.eye(64))
    _, none_norm = build_realign(model, mode="raw", cache_dir=str(tmp_path))
    assert none_norm is None and torch.equal(apply_realign(h, w_id, None), h)


def test_latent_think_extends_cache(tiny, tmp_path):
    model, _ = tiny
    w_a, norm = build_realign(model, cache_dir=str(tmp_path))
    cache, c = new_cache(), Counter()
    latent_think(model, ids(5, 6, 7, 8), cache, 6, w_a, norm, c)
    assert cache.get_seq_length() == 4 + 6
    assert c.forward_passes == 1 + 6 and c.latent_steps == 6 and c.output_tokens == 0
    latent_think(model, ids(9, 10), cache, 3, w_a, norm, c)   # second agent, same cache
    assert cache.get_seq_length() == 10 + 2 + 3


def test_cache_continuation_matches_full_prompt(tiny):
    """Decoding prompt B on top of a cache of prompt A must equal decoding A+B from scratch.
    This catches attention-mask / position-id bugs in the handoff."""
    model, tok = tiny
    a, b = [11, 12, 13, 14, 15], [21, 22, 23]
    c1, cache = Counter(), new_cache()
    prefill(model, ids(*a), cache, c1)
    split = decode(model, tok, ids(*b), cache, c1, max_new_tokens=8)
    full = decode(model, tok, ids(*(a + b)), new_cache(), Counter(), max_new_tokens=8)
    assert split == full


def test_soft_prefix_decode_runs(tiny):
    model, tok = tiny
    prefix = torch.randn(1, 4, 64)
    c = Counter()
    decode(model, tok, ids(3, 4), new_cache(), c, max_new_tokens=5, inputs_embeds_prefix=prefix)
    assert c.forward_passes >= 2


class FakeChatTokenizer(FakeTokenizer):
    """Enough of a tokenizer to run whole pipelines on the tiny model: one id per word,
    ids in [1, 499] so none collide with eos (0) and all fit the tiny vocab."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        word_ids = [1 + sum(map(ord, w)) % 499 for w in text.split()]
        return {"input_ids": torch.tensor([word_ids])}


def test_latent_mas_trace_excludes_judger(tiny, tmp_path):
    """HANDOFF issue 2: cache_len_before_judger must be the latent message size
    (earlier agents' prompts + latent steps), not include the judger's prompt."""
    from types import SimpleNamespace
    from src import pipelines
    from src.modeling import encode_chat
    from src.prompts import PIPELINE, build_messages

    model, _ = tiny
    tok = FakeChatTokenizer()
    w_a, norm = build_realign(model, cache_dir=str(tmp_path))
    steps, question = 3, "Tom has 3 apples and buys 2 more. How many?"
    cfg = SimpleNamespace(max_new_tokens=4, agent_max_tokens=4, latent_steps=steps,
                          temperature=0.0, generator=None)
    r = pipelines.latent_mas(model, tok, question, cfg, w_a, norm)

    agent_prompt_lens = [encode_chat(tok, build_messages(role, question, latent=True), "cpu").shape[1]
                         for role in PIPELINE[:-1]]
    expected = sum(agent_prompt_lens) + steps * len(agent_prompt_lens)
    assert r["trace"]["cache_len_before_judger"] == expected
    assert r["prompt_tokens"] > sum(agent_prompt_lens)   # judger prompt counted separately


def test_dtype_kwarg_matches_transformers_version(tiny, tmp_path, monkeypatch):
    """HANDOFF issue 3: use `dtype=` on new transformers, `torch_dtype=` on old,
    and the loaded weights must actually be in the requested dtype."""
    import transformers
    from transformers import AutoModelForCausalLM
    from src import modeling

    model, _ = tiny
    model.save_pretrained(tmp_path)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, **modeling.dtype_kwarg(torch.float16))
    assert next(loaded.parameters()).dtype == torch.float16

    monkeypatch.setattr(transformers, "__version__", "4.45.0")
    assert modeling.dtype_kwarg(torch.float16) == {"torch_dtype": torch.float16}
    monkeypatch.setattr(transformers, "__version__", "4.56.0")
    assert modeling.dtype_kwarg(torch.float16) == {"dtype": torch.float16}


def test_latent_mas_m0_keep_vs_drop(tiny, tmp_path):
    """HANDOFF issue 6: at m=0, "keep" leaves earlier agents' prompts in the cache;
    "drop" (the reference behaviour) gives the judger an empty cache, so it must
    equal decoding the judger prompt alone from scratch."""
    from types import SimpleNamespace
    from src import pipelines
    from src.modeling import encode_chat
    from src.prompts import build_messages

    model, _ = tiny
    tok = FakeChatTokenizer()
    w_a, norm = build_realign(model, cache_dir=str(tmp_path))
    question = "Tom has 3 apples and buys 2 more. How many?"

    def run(m0_cache):
        cfg = SimpleNamespace(max_new_tokens=6, agent_max_tokens=6, latent_steps=0,
                              temperature=0.0, generator=None, m0_cache=m0_cache)
        return pipelines.latent_mas(model, tok, question, cfg, w_a, norm)

    keep, drop = run("keep"), run("drop")
    assert keep["trace"]["cache_len_before_judger"] > 0
    assert drop["trace"]["cache_len_before_judger"] == 0
    judger_ids = encode_chat(tok, build_messages("judger", question, latent=True), "cpu")
    alone = decode(model, tok, judger_ids, new_cache(), Counter(), max_new_tokens=6)
    assert drop["output"] == alone
    assert drop["forward_passes"] < keep["forward_passes"]


def test_timer_on_cpu():
    from src.pipelines import _Timer
    with _Timer(torch.device("cpu"), torch.device("cpu")) as t:
        pass
    assert t.seconds >= 0 and t.peak_mb == 0.0
