"""Side-by-side demo: the same question through text MAS and latent MAS (HANDOFF task T10).

  python -m scripts.demo                                  # Qwen2.5-3B, opens http://127.0.0.1:7860
  python -m scripts.demo --model Qwen/Qwen2.5-1.5B-Instruct --latent_steps 5
  python -m scripts.demo --precompute 5                   # fill the cache before a live session

Both methods run through the SAME functions the experiment uses
(src/pipelines.text_mas and latent_mas), so the demo cannot drift from what
is measured. It shows one question at a time: it illustrates the mechanism
and the four efficiency metrics; it is NOT evidence for or against H1.

Results are cached in cache/demo_results.json (keyed by model, question and
settings). text MAS takes minutes per question on a laptop, so precompute the
questions you plan to show; a cached question replays instantly.
"""
import argparse
import hashlib
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

from src import pipelines
from src.data import extract_prediction, is_correct, load_gsm8k
from src.modeling import build_realign, default_device, load_model
from src.prompts import PIPELINE

CUSTOM = "(type your own question)"


# ---------------------------------------------------------------- logic (tested)

def make_cfg(latent_steps: int, m0_cache: str = "keep", max_new_tokens: int = 512,
             agent_max_tokens: int = 256) -> SimpleNamespace:
    """Same config object scripts/run.py builds; greedy decoding, as in the experiment."""
    return SimpleNamespace(max_new_tokens=max_new_tokens, agent_max_tokens=agent_max_tokens,
                           latent_steps=latent_steps, temperature=0.0, generator=None,
                           m0_cache=m0_cache)


def score(result: Dict, gold: Optional[str]) -> Dict:
    """Attach pred/correct with the experiment's one scorer (correct is None without gold)."""
    pred = extract_prediction(result["output"])
    return {**result, "pred": pred, "correct": is_correct(pred, gold) if gold else None}


def _ratio(new: float, old: float) -> str:
    if not old:
        return "–"
    return f"{(new - old) / old:+.0%}"


def comparison_rows(text_r: Optional[Dict], latent_r: Optional[Dict]) -> List[List[str]]:
    """The metrics table: one row per metric, text vs latent, and the relative change.
    Forward passes and latency sit next to output tokens on purpose (invariant 5):
    each latent step is a full forward pass, so tokens alone flatter latent MAS."""
    def cell(r, key, fmt="{}"):
        return "…" if r is None else fmt.format(r[key])

    def ok(r):
        if r is None:
            return "…"
        mark = {True: "✅", False: "❌", None: ""}[r["correct"]]
        return f"{r['pred']} {mark}".strip() if r["pred"] is not None else f"(none) {mark}".strip()

    rows = [
        ["Answer", ok(text_r), ok(latent_r), ""],
        ["Output tokens", cell(text_r, "output_tokens"), cell(latent_r, "output_tokens")],
        ["Forward passes", cell(text_r, "forward_passes"), cell(latent_r, "forward_passes")],
        ["  of which latent steps", cell(text_r, "latent_steps"), cell(latent_r, "latent_steps")],
        ["Prompt tokens", cell(text_r, "prompt_tokens"), cell(latent_r, "prompt_tokens")],
        ["Latency (s)", cell(text_r, "latency_s", "{:.1f}"), cell(latent_r, "latency_s", "{:.1f}")],
    ]
    keys = [None, "output_tokens", "forward_passes", None, "prompt_tokens", "latency_s"]
    for row, key in zip(rows[1:], keys[1:]):
        if key and text_r is not None and latent_r is not None:
            row.append(_ratio(latent_r[key], text_r[key]))
        else:
            row.append("")
    return rows


def text_transcript(text_r: Dict) -> str:
    """Markdown of what each text agent wrote; this is exactly what was pasted forward."""
    parts = []
    for role, n in zip(PIPELINE[:-1], text_r["per_agent_tokens"]):
        parts.append(f"#### {role.title()} · {n} tokens\n\n{text_r['trace'][role]}")
    return "\n\n---\n\n".join(parts)


def latent_message(latent_r: Dict, latent_steps: int) -> str:
    n = latent_r["trace"]["cache_len_before_judger"]
    if n == 0:
        return ("**Message to the judger:** nothing. With `m=0` and `drop`, the judger starts "
                "from an empty cache.")
    return (f"**Message to the judger:** a KV cache of **{n} positions**: the three agents' "
            f"prompts plus {3 * latent_steps} latent steps ({latent_steps} per agent). "
            "No text was written; these thoughts are not human-readable.")


def cache_key(model_name: str, question: str, cfg: SimpleNamespace, realign: str) -> str:
    blob = json.dumps([model_name, question, cfg.latent_steps, realign, cfg.m0_cache,
                       cfg.max_new_tokens, cfg.agent_max_tokens])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


class ResultCache:
    """Tiny JSON cache so a live demo can replay precomputed questions instantly."""

    def __init__(self, path: str):
        self.path = path
        self.data = {}
        if os.path.exists(path):
            with open(path) as f:
                self.data = json.load(f)

    def get(self, key: str, method: str) -> Optional[Dict]:
        return self.data.get(key, {}).get(method)

    def put(self, key: str, method: str, result: Dict):
        self.data.setdefault(key, {})[method] = result
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f)


class Demo:
    """Holds the model and realign matrices; runs one method with caching."""

    def __init__(self, model, tok, model_name: str, cache: ResultCache, defaults: Dict):
        self.model, self.tok, self.model_name = model, tok, model_name
        self.cache, self.defaults = cache, defaults
        self._realign = {}

    def realign(self, mode: str):
        if mode not in self._realign:
            self._realign[mode] = build_realign(self.model, mode=mode)
        return self._realign[mode]

    def run(self, method: str, question: str, gold: Optional[str], cfg, realign: str,
            use_cache: bool = True) -> Dict:
        # text MAS does not depend on m / realign; key it on defaults so it is reused.
        key_cfg = cfg if method == "latent_mas" else make_cfg(0, "keep", cfg.max_new_tokens,
                                                              cfg.agent_max_tokens)
        key = cache_key(self.model_name, question, key_cfg,
                        realign if method == "latent_mas" else "-")
        if use_cache and (hit := self.cache.get(key, method)) is not None:
            return score({**hit, "cached": True}, gold)
        if method == "latent_mas":
            w_a, norm = self.realign(realign)
            r = pipelines.latent_mas(self.model, self.tok, question, cfg, w_a, norm)
        else:
            r = pipelines.text_mas(self.model, self.tok, question, cfg)
        self.cache.put(key, method, r)
        return score({**r, "cached": False}, gold)


# ---------------------------------------------------------------- UI

def build_ui(demo: Demo, examples: List[Dict], default_m: int, default_realign: str):
    import gradio as gr  # imported here so the logic above is testable without gradio

    by_label = {f"{it['id']}: {it['question'][:70]}…": it for it in examples}
    labels = [CUSTOM] + list(by_label)
    latex = [{"left": "$$", "right": "$$", "display": True},
             {"left": "\\[", "right": "\\]", "display": True},
             {"left": "\\(", "right": "\\)", "display": False}]

    def pick(label):
        it = by_label.get(label)
        return (it["question"], it["gold"]) if it else ("", "")

    def compare(question, gold, m, realign, m0_cache, use_cache):
        question, gold = question.strip(), (gold or "").strip() or None
        if not question:
            raise gr.Error("Pick an example or type a question.")
        cfg = make_cfg(int(m), m0_cache, demo.defaults["max_new_tokens"],
                       demo.defaults["agent_max_tokens"])
        if int(m) != 0 and m0_cache == "drop":
            cfg.m0_cache = "keep"   # drop only means something at m = 0

        status = "Running **latent MAS**…"
        yield (comparison_rows(None, None), "", "", "", "", status)
        lat = demo.run("latent_mas", question, gold, cfg, realign, use_cache)
        lat_md = latent_message(lat, int(m))
        status = ("Latent MAS done" + (" (cached)" if lat["cached"] else "") +
                  ". Running **text MAS** (the slow one: every agent writes text)…")
        yield (comparison_rows(None, lat), "", "", lat["output"], lat_md, status)

        txt = demo.run("text_mas", question, gold, cfg, realign, use_cache)
        status = "Done" + (" (both cached)" if lat["cached"] and txt["cached"] else "") + "."
        yield (comparison_rows(txt, lat), text_transcript(txt), txt["output"],
               lat["output"], lat_md, status)

    with gr.Blocks(title="Unspoken: text vs latent MAS") as ui:
        gr.Markdown(
            "# Unspoken: text vs latent multi-agent reasoning\n"
            "Planner → critic → refiner → judger on the same question. **Text MAS** passes "
            "written messages between agents; **latent MAS** passes the KV cache and only the "
            "judger writes. Same model, same prompts, same decoding loop.\n\n"
            f"Model: `{demo.model_name}` · greedy decoding · "
            "*One question is an illustration, not evidence: H1 is tested by `scripts.analyze`.*")
        with gr.Row():
            with gr.Column(scale=3):
                example = gr.Dropdown(labels, value=labels[1] if len(labels) > 1 else CUSTOM,
                                      label="GSM8K test question")
                question = gr.Textbox(label="Question", lines=4)
                gold = gr.Textbox(label="Gold answer (optional, for scoring)", max_lines=1)
            with gr.Column(scale=2):
                m = gr.Slider(0, 40, value=default_m, step=1, label="Latent steps per agent (m)",
                              info="Experiment default: 10. Qwen2.5-3B stays coherent at about 1–5.")
                realign = gr.Radio(["full", "norm_only", "raw"], value=default_realign,
                                   label="Realign mode")
                m0_cache = gr.Radio(["keep", "drop"], value="keep", label="At m=0: cache",
                                    info="drop = reference semantics (judger gets no cache)")
                use_cache = gr.Checkbox(value=True, label="Reuse cached results")
                run = gr.Button("Compare", variant="primary")
        status = gr.Markdown()
        table = gr.Dataframe(headers=["Metric", "Text MAS", "Latent MAS", "Latent vs text"],
                             value=comparison_rows(None, None), interactive=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Text MAS: judger")
                text_out = gr.Markdown(latex_delimiters=latex)
                with gr.Accordion("Messages passed between agents (text)", open=False):
                    transcript = gr.Markdown(latex_delimiters=latex)
            with gr.Column():
                gr.Markdown("### Latent MAS: judger")
                latent_out = gr.Markdown(latex_delimiters=latex)
                latent_msg = gr.Markdown()

        example.change(pick, example, [question, gold])
        ui.load(pick, example, [question, gold])
        run.click(compare, [question, gold, m, realign, m0_cache, use_cache],
                  [table, transcript, text_out, latent_out, latent_msg, status])
    return ui


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--latent_steps", type=int, default=10, help="initial slider value")
    p.add_argument("--realign", default="full", choices=["full", "norm_only", "raw"])
    p.add_argument("--n_examples", type=int, default=20, help="GSM8K test questions in the dropdown")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--agent_max_tokens", type=int, default=256)
    p.add_argument("--device", default=None, help="default: cuda > mps > cpu")
    p.add_argument("--cache", default="cache/demo_results.json")
    p.add_argument("--precompute", type=int, default=0,
                   help="run the first N examples at --latent_steps, fill the cache, and exit")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="public gradio.live link (e.g. from Kaggle)")
    args = p.parse_args()

    device = args.device or default_device()
    model, tok = load_model(args.model, device)
    print(f"model={args.model} device={device} tie_word_embeddings={model.config.tie_word_embeddings}")
    demo = Demo(model, tok, args.model, ResultCache(args.cache),
                {"max_new_tokens": args.max_new_tokens, "agent_max_tokens": args.agent_max_tokens})
    examples = load_gsm8k("test", args.n_examples)

    if args.precompute:
        cfg = make_cfg(args.latent_steps, "keep", args.max_new_tokens, args.agent_max_tokens)
        for it in examples[:args.precompute]:
            for method in ("latent_mas", "text_mas"):
                r = demo.run(method, it["question"], it["gold"], cfg, args.realign)
                took = "cached" if r["cached"] else f"{r['latency_s']:.0f}s"
                print(f"{it['id']} {method:10s} pred={r['pred']} correct={r['correct']} "
                      f"tokens={r['output_tokens']} fwd={r['forward_passes']} {took}")
        return

    torch.set_grad_enabled(False)
    build_ui(demo, examples, args.latent_steps, args.realign).queue().launch(
        server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
