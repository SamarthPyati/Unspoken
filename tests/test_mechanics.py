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


def test_demo_runs_both_methods_and_caches(tiny, tmp_path, monkeypatch):
    """scripts/demo.py: runs the experiment's own pipelines, scores with the one scorer,
    and a cached replay returns the identical result."""
    from scripts import demo as d
    import src.modeling as modeling

    model, _ = tiny
    monkeypatch.setattr(d, "build_realign",
                        lambda m, mode: modeling.build_realign(m, mode=mode, cache_dir=str(tmp_path)))
    app = d.Demo(model, FakeChatTokenizer(), "tiny", d.ResultCache(str(tmp_path / "c.json")),
                 {"max_new_tokens": 4, "agent_max_tokens": 4})
    cfg = d.make_cfg(2, max_new_tokens=4, agent_max_tokens=4)
    q = "Tom has 3 apples and buys 2 more. How many?"
    lat = app.run("latent_mas", q, "5", cfg, "full")
    txt = app.run("text_mas", q, "5", cfg, "full")
    assert not lat["cached"] and lat["latent_steps"] == 2 * 3 and lat["correct"] in (True, False)
    assert txt["latent_steps"] == 0 and len(txt["per_agent_tokens"]) == 4

    again = d.Demo(model, FakeChatTokenizer(), "tiny", d.ResultCache(str(tmp_path / "c.json")),
                   app.defaults).run("latent_mas", q, "5", cfg, "full")
    assert again["cached"] and again["output"] == lat["output"]

    rows = d.comparison_rows(txt, lat)
    assert [r[0] for r in rows][:3] == ["Answer", "Output tokens", "Forward passes"]
    assert all(len(r) == 4 for r in rows)
    assert "per agent" in d.latent_message(lat, 2)
    assert "Planner" in d.text_transcript(txt)


def test_demo_ui_builds(tiny, tmp_path):
    gr = pytest.importorskip("gradio")
    from scripts import demo as d

    model, _ = tiny
    app = d.Demo(model, FakeChatTokenizer(), "tiny", d.ResultCache(str(tmp_path / "c.json")),
                 {"max_new_tokens": 4, "agent_max_tokens": 4})
    ui = d.build_ui(app, [{"id": "x-0", "question": "What is 2+3?", "gold": "5"}], 5, "full")
    assert isinstance(ui, gr.Blocks)


def test_callbacks_only_observe(tiny, tmp_path):
    """The demo's streaming hooks must not change outputs or counts, and must see every
    generated token and every latent step (invariant: one loop for all methods)."""
    from types import SimpleNamespace
    from src import pipelines

    model, _ = tiny
    tok = FakeChatTokenizer()
    w_a, norm = build_realign(model, cache_dir=str(tmp_path))
    q = "Tom has 3 apples and buys 2 more. How many?"
    base = dict(max_new_tokens=5, agent_max_tokens=5, latent_steps=2, temperature=0.0,
                generator=None, m0_cache="keep")
    for name, run in (("text_mas", lambda cfg: pipelines.text_mas(model, tok, q, cfg)),
                      ("latent_mas", lambda cfg: pipelines.latent_mas(model, tok, q, cfg, w_a, norm))):
        events = []
        plain = run(SimpleNamespace(**base))
        hooked = run(SimpleNamespace(**base, callback=lambda role, ev, v: events.append((role, ev))))
        for k in ("output", "output_tokens", "forward_passes", "latent_steps", "per_agent_tokens"):
            assert plain[k] == hooked[k], (name, k)
        assert sum(ev == "token" for _, ev in events) == hooked["output_tokens"]
        assert sum(ev == "latent_step" for _, ev in events) == hooked["latent_steps"]


def test_demo_chat_streams_and_renders(tiny, tmp_path, monkeypatch):
    """Chat demo: streaming yields live bubbles and then the same result a plain run gives;
    final bubbles carry the metrics; the scoreboard and session totals add up."""
    from scripts import demo as d
    import src.modeling as modeling

    model, _ = tiny
    monkeypatch.setattr(d, "build_realign",
                        lambda m, mode: modeling.build_realign(m, mode=mode, cache_dir=str(tmp_path)))
    tok = FakeChatTokenizer()
    app = d.Demo(model, tok, "tiny", d.ResultCache(str(tmp_path / "c.json")),
                 {"max_new_tokens": 4, "agent_max_tokens": 4})
    cfg = d.make_cfg(2, max_new_tokens=4, agent_max_tokens=4)
    q = "Tom has 3 apples and buys 2 more. How many?"

    frames = list(d._stream(app, "text_mas", q, "5", cfg, "full", False,
                            lambda live, r: d.text_messages(tok, live, r)))
    text_r, final = frames[-1]
    assert text_r is not None and all(res is None for res, _ in frames[:-1])
    assert [m["metadata"]["title"].split()[1] for m in final[:3]] == ["Planner", "Critic", "Refiner"]
    assert "tokens written" in final[-1]["content"] and "metadata" not in final[-1]
    plain = app.run("text_mas", q, "5", cfg, "full", use_cache=False)
    assert plain["output"] == text_r["output"] and plain["output_tokens"] == text_r["output_tokens"]

    lat_r, lat_final = list(d._stream(app, "latent_mas", q, "5", cfg, "full", False,
                                      lambda live, r: d.latent_messages(tok, live, r, 2)))[-1]
    assert "6 latent steps, 0 tokens" in lat_final[0]["metadata"]["title"]
    live = d.latent_messages(tok, {"_steps": 3}, None, 2)
    assert live[0]["metadata"]["status"] == "pending" and "3 of 6" in live[0]["content"]

    totals = d.add_to_totals({}, text_r, lat_r)
    assert totals["n"] == 1 and totals["text_tok"] == text_r["output_tokens"] and totals["scored"] == 1
    html = d.scoreboard(text_r, lat_r, totals)
    assert "Tokens written" in html and "Session · 1 question" in html
    assert "Ask a question" in d.scoreboard(None, None, {})


def test_demo_pretty_boxed_display_only():
    from scripts.demo import pretty
    assert pretty("Final Answer:\n\n\\boxed{3}") == "Final Answer:\n\n\\[\\boxed{3}\\]"
    inline = "so \\(\\boxed{18}\\) dollars"
    block = "\\[\n\\boxed{36}\n\\]"
    assert pretty(inline) == inline and pretty(block) == block   # already math: untouched


def test_demo_stop_aborts_and_caches_nothing(tiny, tmp_path, monkeypatch):
    import threading
    from scripts import demo as d
    import src.modeling as modeling

    model, _ = tiny
    monkeypatch.setattr(d, "build_realign",
                        lambda m, mode: modeling.build_realign(m, mode=mode, cache_dir=str(tmp_path)))
    tok = FakeChatTokenizer()
    cache = d.ResultCache(str(tmp_path / "c.json"))
    app = d.Demo(model, tok, "tiny", cache, {"max_new_tokens": 4, "agent_max_tokens": 4})
    cfg = d.make_cfg(2, max_new_tokens=4, agent_max_tokens=4)
    q = "Tom has 3 apples and buys 2 more. How many?"

    flag = threading.Event(); flag.set()
    for method in ("text_mas", "latent_mas"):
        with pytest.raises(d.Interrupted):
            list(d._stream(app, method, q, "5", cfg, "full", True, lambda live, r: [], flag))
    assert cache.data == {}                                   # nothing partial was cached

    live = {"planner": [5, 6, 7], "_current": "planner"}
    frozen = d.stopped(d.text_messages(tok, live, None), live)
    assert frozen[0]["metadata"]["status"] == "done"
    assert "Stopped** after 3 tokens" in frozen[-1]["content"]

    ok = list(d._stream(app, "text_mas", q, "5", cfg, "full", True, lambda live, r: [], threading.Event()))
    assert ok[-1][0] is not None and len(cache.data) == 1      # a later normal run still works
