"""Chat demo: every question goes to text agents and to latent agents, side by side (HANDOFF T10).

  python -m scripts.demo                                  # Qwen2.5-3B, opens http://127.0.0.1:7860
  python -m scripts.demo --model Qwen/Qwen2.5-1.5B-Instruct
  python -m scripts.demo --precompute 5                   # fill the cache before a live session

Left: the four agents talk in text, and you watch the planner, critic and refiner write their
messages. Right: the same agents share their working memory (KV cache), so only the final agent
writes. A scoreboard keeps the tokens and time saved, per question and over the session.

Both columns run through the SAME functions the experiment uses (src/pipelines.text_mas and
latent_mas) and the same scorer, so the demo cannot drift from what is measured. Text is streamed
through an observe-only hook (tested to leave outputs unchanged). Each question is answered on its
own; the chat does not feed earlier turns back in, because that would change the prompts.
One question illustrates the mechanism; it is NOT evidence for or against H1.

Results are cached in cache/demo_results.json, so a precomputed question replays instantly.
"""
import argparse
import hashlib
import json
import os
import queue
import re
import threading
import time
from types import SimpleNamespace
from typing import Dict, List, Optional

import torch

from src import pipelines
from src.data import extract_prediction, is_correct, load_gsm8k
from src.modeling import build_realign, default_device, load_model
from src.prompts import PIPELINE

AGENTS = PIPELINE[:-1]            # planner, critic, refiner: the agents that talk to each other


class Interrupted(Exception):
    """Raised from the streaming hook when the user presses Stop. It aborts generation at the
    next token or latent step, so the GPU is freed at once; nothing partial is cached."""

    def __init__(self, live: Optional[Dict] = None):
        super().__init__("stopped by the user")
        self.live = live or {}
LATEX = [{"left": "$$", "right": "$$", "display": True},
         {"left": "\\[", "right": "\\]", "display": True},
         {"left": "\\(", "right": "\\)", "display": False}]


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
    for role, n in zip(AGENTS, text_r["per_agent_tokens"]):
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
        self.stops: Dict[str, threading.Event] = {}   # Stop flag per browser session

    def realign(self, mode: str):
        if mode not in self._realign:
            self._realign[mode] = build_realign(self.model, mode=mode)
        return self._realign[mode]

    def run(self, method: str, question: str, gold: Optional[str], cfg, realign: str,
            use_cache: bool = True, callback=None) -> Dict:
        """callback(role, event, value) observes generation live (see pipelines._hooks)."""
        # text MAS does not depend on m / realign; key it on defaults so it is reused.
        key_cfg = cfg if method == "latent_mas" else make_cfg(0, "keep", cfg.max_new_tokens,
                                                              cfg.agent_max_tokens)
        key = cache_key(self.model_name, question, key_cfg,
                        realign if method == "latent_mas" else "-")
        if use_cache and (hit := self.cache.get(key, method)) is not None:
            return score({**hit, "cached": True}, gold)
        run_cfg = SimpleNamespace(**vars(cfg), callback=callback)
        if method == "latent_mas":
            w_a, norm = self.realign(realign)
            r = pipelines.latent_mas(self.model, self.tok, question, run_cfg, w_a, norm)
        else:
            r = pipelines.text_mas(self.model, self.tok, question, run_cfg)
        self.cache.put(key, method, r)
        return score({**r, "cached": False}, gold)


# ---------------------------------------------------------------- chat rendering (tested)

_BARE_BOX = re.compile(r"^\s*\\boxed\{[^{}]*\}\s*$")


def pretty(text: str) -> str:
    """Display only: a \\boxed{} alone on a line and outside any math block is wrapped in
    \\[ \\] so it renders. Scoring always uses the raw output."""
    out, in_math = [], False
    for line in text.split("\n"):
        t = line.strip()
        if t in ("\\[", "$$") and not in_math:
            in_math = True
        elif t in ("\\]", "$$") and in_math:
            in_math = False
        elif not in_math and _BARE_BOX.match(line):
            line = f"\\[{t}\\]"
        out.append(line)
    return "\n".join(out)


def footer(r: Dict) -> str:
    mark = {True: " · ✅ correct", False: " · ❌ wrong", None: ""}[r["correct"]]
    cached = " · replayed from cache" if r.get("cached") else ""
    return (f"\n\n<sub>**{r['output_tokens']} tokens written** · {r['forward_passes']} forward "
            f"passes · {r['latency_s']:.1f} s{mark}{cached}</sub>")


def text_messages(tok, live: Dict, result: Optional[Dict]) -> List[Dict]:
    """Assistant messages for the text column: one collapsible bubble per talking agent,
    then the judger's answer. `live` holds token ids per role while generating."""
    msgs = []
    for role in AGENTS:
        if result is not None:
            text, n, done = result["trace"][role], result["per_agent_tokens"][AGENTS.index(role)], True
        elif role in live:
            text, n = tok.decode(live[role], skip_special_tokens=True), len(live[role])
            done = live.get("_current") != role
        else:
            continue
        msgs.append({"role": "assistant", "content": text,
                     "metadata": {"title": f"📝 {role.title()} wrote {n} tokens to the next agent",
                                  "status": "done" if done else "pending"}})
    if result is not None:
        msgs.append({"role": "assistant", "content": pretty(result["output"]) + footer(result)})
    elif "judger" in live:
        msgs.append({"role": "assistant", "content": tok.decode(live["judger"], skip_special_tokens=True)})
    return msgs


def latent_messages(tok, live: Dict, result: Optional[Dict], m: int) -> List[Dict]:
    """Assistant messages for the latent column: one bubble for the silent thinking,
    then the judger's answer."""
    steps = result["latent_steps"] if result is not None else live.get("_steps", 0)
    done = result is not None or "judger" in live
    body = (latent_message(result, m) if result is not None else
            f"{steps} of {3 * m} latent steps done · 0 tokens written")
    msgs = [{"role": "assistant", "content": body,
             "metadata": {"title": f"🧠 Planner → critic → refiner thought silently: "
                                   f"{steps} latent steps, 0 tokens",
                          "status": "done" if done else "pending"}}]
    if result is not None:
        msgs.append({"role": "assistant", "content": pretty(result["output"]) + footer(result)})
    elif "judger" in live:
        msgs.append({"role": "assistant", "content": tok.decode(live["judger"], skip_special_tokens=True)})
    return msgs


def stopped(msgs: List[Dict], live: Dict) -> List[Dict]:
    """Freeze a column after Stop: close pending bubbles, add a note. Partial text stays visible."""
    n = sum(len(v) for k, v in live.items() if not k.startswith("_"))
    out = [{**m, "metadata": {**m["metadata"], "status": "done"}} if m.get("metadata") else m
           for m in msgs]
    out.append({"role": "assistant",
                "content": f"⏹ **Stopped** after {n} tokens written. Partial answers are not cached or scored."})
    return out


def scoreboard(text_r: Optional[Dict], lat_r: Optional[Dict], totals: Dict) -> str:
    """HTML stat cards: this question, then the whole session."""
    def card(label, value, sub):
        return (f"<div class='stat'><div class='label'>{label}</div><div class='value'>{value}</div>"
                f"<div class='sub'>{sub}</div></div>")

    cards = []
    if text_r and lat_r:
        saved = 1 - lat_r["output_tokens"] / max(1, text_r["output_tokens"])
        speed = text_r["latency_s"] / max(1e-6, lat_r["latency_s"])
        fwd = 1 - lat_r["forward_passes"] / max(1, text_r["forward_passes"])
        cards += [card("Tokens written", f"−{saved:.0%}",
                       f"{text_r['output_tokens']} → {lat_r['output_tokens']}"),
                  card("Answer time", f"{speed:.1f}× faster",
                       f"{text_r['latency_s']:.1f} s → {lat_r['latency_s']:.1f} s"),
                  card("Forward passes", f"−{fwd:.0%}",
                       f"{text_r['forward_passes']} → {lat_r['forward_passes']}")]
    if totals.get("n"):
        t, l = totals["text_tok"], totals["lat_tok"]
        acc = ""
        if totals["scored"]:
            acc = f"correct: text {totals['text_ok']}/{totals['scored']}, latent {totals['lat_ok']}/{totals['scored']}"
        cards.append(card(f"Session · {totals['n']} question{'s' * (totals['n'] > 1)}",
                          f"−{1 - l / max(1, t):.0%} tokens", f"{t} → {l} tokens. {acc}"))
    if not cards:
        return "<div class='stats empty'>Ask a question to see how many tokens the latent agents save.</div>"
    return "<div class='stats'>" + "".join(cards) + "</div>"


def add_to_totals(totals: Dict, text_r: Dict, lat_r: Dict) -> Dict:
    t = dict(totals or {"n": 0, "text_tok": 0, "lat_tok": 0, "scored": 0, "text_ok": 0, "lat_ok": 0})
    t["n"] += 1
    t["text_tok"] += text_r["output_tokens"]
    t["lat_tok"] += lat_r["output_tokens"]
    if text_r["correct"] is not None:
        t["scored"] += 1
        t["text_ok"] += bool(text_r["correct"])
        t["lat_ok"] += bool(lat_r["correct"])
    return t


# ---------------------------------------------------------------- UI

CSS = """
.stats {display: flex; gap: 12px; flex-wrap: wrap; margin: 4px 0 8px;}
.stats.empty {opacity: .7; font-size: .95em; padding: 8px 2px;}
.stat {flex: 1 1 160px; border: 1px solid var(--border-color-primary); border-radius: 12px;
       padding: 10px 14px; background: var(--background-fill-secondary);}
.stat .label {font-size: .8em; opacity: .75; text-transform: uppercase; letter-spacing: .04em;}
.stat .value {font-size: 1.6em; font-weight: 700; margin: 2px 0;}
.stat .sub {font-size: .85em; opacity: .8;}
.colhead {font-weight: 600; margin-bottom: -6px;}
"""


def _stream(demo: Demo, method: str, question: str, gold, cfg, realign, use_cache, render,
            stop: Optional[threading.Event] = None):
    """Run one method in a thread; yield render(live, None) while it generates, then
    render(None, result). The model runs one request at a time (Gradio queue).
    If `stop` is set, generation is aborted at the next token and Interrupted(live) is raised."""
    events, box = queue.Queue(), {}

    def cb(role, event, value):
        if stop is not None and stop.is_set():
            raise Interrupted()           # unwinds decode/latent_think: the model stops now
        events.put((role, event, value))

    def work():
        try:
            box["r"] = demo.run(method, question, gold, cfg, realign, use_cache, callback=cb)
        except Exception as exc:        # surface errors in the UI instead of hanging
            box["err"] = exc

    th = threading.Thread(target=work, daemon=True)
    th.start()
    live, last = {}, 0.0
    while th.is_alive() or not events.empty():
        try:
            role, event, value = events.get(timeout=0.1)
            if event == "token":
                live.setdefault(role, []).append(value)
                live["_current"] = role
            else:
                live["_steps"] = live.get("_steps", 0) + 1
        except queue.Empty:
            pass
        if time.time() - last > 0.15:
            last = time.time()
            yield None, render(live, None)
    th.join()
    if isinstance(box.get("err"), Interrupted):
        raise Interrupted(live)
    if "err" in box:
        raise box["err"]
    yield box["r"], render(None, box["r"])


def _session(request) -> str:
    return getattr(request, "session_hash", None) or "local"


def build_ui(demo: Demo, examples: List[Dict], default_m: int, default_realign: str):
    import gradio as gr  # imported here so the logic above is testable without gradio

    gold_by_q = {it["question"]: it["gold"] for it in examples}

    def respond(message, text_hist, lat_hist, totals, m, realign, use_cache, request: gr.Request = None):
        q = (message or "").strip()
        if not q:
            raise gr.Error("Type a math word problem, or pick an example below.")
        flag = demo.stops[_session(request)] = threading.Event()   # one Stop flag per browser session
        show, hide = gr.update(visible=True), gr.update(visible=False)
        m = int(m)
        cfg = make_cfg(m, "keep", demo.defaults["max_new_tokens"], demo.defaults["agent_max_tokens"])
        gold = gold_by_q.get(q)
        user = {"role": "user", "content": q}
        text_base, lat_base = text_hist + [user], lat_hist + [user]
        waiting = [{"role": "assistant", "content": "Waiting: runs after the latent agents, on the same GPU.",
                    "metadata": {"title": "⏳ Text agents", "status": "pending"}}]
        board = scoreboard(None, None, totals)

        skipped = [{"role": "assistant", "content": "Not run: stopped before the text agents started."}]

        lat_r = None
        render_lat = lambda live, r: latent_messages(demo.tok, live, r, m)
        try:
            for res, msgs in _stream(demo, "latent_mas", q, gold, cfg, realign, use_cache, render_lat, flag):
                lat_r = res or lat_r
                yield "", text_base + waiting, lat_base + msgs, totals, board, show
        except Interrupted as e:
            yield "", text_base + skipped, lat_base + stopped(render_lat(e.live, None), e.live), totals, board, hide
            return
        lat_final = lat_base + msgs

        text_r = None
        render_txt = lambda live, r: text_messages(demo.tok, live, r)
        try:
            for res, msgs in _stream(demo, "text_mas", q, gold, cfg, realign, use_cache, render_txt, flag):
                text_r = res or text_r
                yield "", text_base + msgs, lat_final, totals, board, show
        except Interrupted as e:
            yield "", text_base + stopped(render_txt(e.live, None), e.live), lat_final, totals, board, hide
            return
        totals = add_to_totals(totals, text_r, lat_r)
        yield "", text_base + msgs, lat_final, totals, scoreboard(text_r, lat_r, totals), hide

    def stop_generation(request: gr.Request = None):
        flag = demo.stops.get(_session(request))
        if flag is not None:
            flag.set()

    with gr.Blocks(title="Unspoken: agents that think without talking") as ui:
        gr.Markdown(
            "# Unspoken: agents that think without talking\n"
            "Four agents (planner → critic → refiner → judger) solve each question twice, with the "
            "same model and the same prompts. **Left:** they write messages to each other. "
            "**Right:** they share working memory (the KV cache) instead, and only the final agent "
            f"writes. Model: `{demo.model_name}`.")
        board = gr.HTML(scoreboard(None, None, {}))
        totals = gr.State({})
        with gr.Row(equal_height=True):
            with gr.Column():
                gr.Markdown("📝 **Text agents** (baseline)", elem_classes="colhead")
                text_chat = gr.Chatbot(height=520, latex_delimiters=LATEX, show_label=False,
                                       placeholder="The agents' messages appear here.")
            with gr.Column():
                gr.Markdown("🧠 **Unspoken** (latent agents)", elem_classes="colhead")
                lat_chat = gr.Chatbot(height=520, latex_delimiters=LATEX, show_label=False,
                                      placeholder="Only the final answer is written here.")
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask a math word problem…", show_label=False, scale=8,
                             autofocus=True, submit_btn=True)
            stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1, visible=False)
            clear = gr.Button("Clear", scale=1)
        gr.Examples([[it["question"]] for it in examples[:6]], inputs=[msg],
                    label="GSM8K examples (answers are scored against the gold answer)")
        with gr.Accordion("Settings", open=False):
            m = gr.Slider(0, 40, value=default_m, step=1, label="Latent steps per agent (m)",
                          info="On Qwen2.5-3B only m≈1 stays coherent; the experiment's pre-registered "
                               "default is 10 (HANDOFF issue 4).")
            realign = gr.Radio(["full", "norm_only", "raw"], value=default_realign, label="Realign mode")
            use_cache = gr.Checkbox(value=True, label="Replay cached answers when available")
        gr.Markdown("<sub>Each question is answered on its own: earlier turns are not fed back in, so "
                    "the prompts stay identical to the experiment's. One question is an illustration, "
                    "not evidence; H1 is tested on 1,319 problems by `scripts.analyze`.</sub>")

        outputs = [msg, text_chat, lat_chat, totals, board, stop_btn]
        msg.submit(respond, [msg, text_chat, lat_chat, totals, m, realign, use_cache], outputs)
        # queue=False: Stop must act immediately, not wait behind the running request.
        stop_btn.click(stop_generation, None, None, queue=False)
        clear.click(lambda: ("", [], [], {}, scoreboard(None, None, {}), gr.update(visible=False)),
                    None, outputs)
    return ui


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--latent_steps", type=int, default=1,
                   help="initial m (1: the setting that is coherent on Qwen2.5-3B)")
    p.add_argument("--realign", default="full", choices=["full", "norm_only", "raw"])
    p.add_argument("--n_examples", type=int, default=20, help="GSM8K test questions offered as examples")
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

    import gradio as gr
    torch.set_grad_enabled(False)
    build_ui(demo, examples, args.latent_steps, args.realign).queue().launch(
        server_port=args.port, share=args.share, theme=gr.themes.Soft(), css=CSS)


if __name__ == "__main__":
    main()
