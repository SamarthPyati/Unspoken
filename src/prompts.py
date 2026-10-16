"""Agent prompts.

The text and latent pipelines use the SAME role instructions. The only
difference is where the previous agents' work comes from:
  - text mode:   pasted into the prompt as text (`context`)
  - latent mode: already sitting in the KV cache, so `context` is empty
Keeping everything else identical is what makes the comparison fair.
"""
from typing import List, Dict

SYSTEM = "You are a helpful assistant."

ROLE_INSTRUCTIONS = {
    "planner": (
        "You are a Planner Agent. Design a concise, step-by-step plan to solve the "
        "question. Do not compute the final answer."
    ),
    "critic": (
        "You are a Critic Agent. Check the previous plan for mistakes or missing steps "
        "and give brief, constructive feedback."
    ),
    "refiner": (
        "You are a Refiner Agent. Using the plan and the feedback, write a corrected, "
        "concise step-by-step plan."
    ),
    "judger": (
        "Solve the question. Reason step by step, then give the final answer inside "
        "\\boxed{}."
    ),
}

PIPELINE = ["planner", "critic", "refiner", "judger"]


def build_messages(role: str, question: str, context: str = "", latent: bool = False) -> List[Dict]:
    parts = [ROLE_INSTRUCTIONS[role], f"Question: {question}"]
    if latent and role != "planner":
        parts.append("Earlier agents' work is provided to you in latent form.")
    if context:
        parts.append(f"Earlier agents' work:\n{context}")
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def single_agent_messages(question: str) -> List[Dict]:
    """Baseline 1: one model, plain chain-of-thought."""
    return build_messages("judger", question)
