"""typed-decisions demo: BANKING77 intent in a few milliseconds, with calibrated probabilities.

Loads the published linear probe and its base embedding model, embeds the query on CPU, and
shows the top intents with probabilities and the measured latency of this request.
"""

import json
import os
import time

import gradio as gr
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

REPO = os.environ.get("PROBE_REPO", "<hf-user>/banking77-intent-probe-minilm")

cfg = json.load(open(hf_hub_download(REPO, "config.json")))
probe = load_file(hf_hub_download(REPO, "probe.safetensors"))
tok = AutoTokenizer.from_pretrained(cfg["base_model"])
enc = AutoModel.from_pretrained(cfg["base_model"]).eval()
torch.set_num_threads(max(1, os.cpu_count() or 1))


@torch.inference_mode()
def classify(text: str):
    start = time.perf_counter_ns()
    batch = tok([text], return_tensors="pt", truncation=True, max_length=256)
    h = enc(**batch).last_hidden_state
    m = batch["attention_mask"].unsqueeze(-1)
    v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=1)
    p = torch.softmax(v @ probe["weight"] * cfg["logit_scale"] + probe["bias"], dim=1)[0]
    ms = (time.perf_counter_ns() - start) / 1e6
    top = torch.topk(p, 5)
    labels = {cfg["labels"][int(i)].replace("_", " "): float(s) for s, i in zip(top.values, top.indices)}
    return labels, f"{ms:.1f} ms on this Space's CPU"


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="A banking customer's message",
                      value="I was charged twice for the same purchase"),
    outputs=[gr.Label(label="Intent (calibrated probability)"), gr.Textbox(label="Latency")],
    title="typed-decisions: 77-way intent in milliseconds",
    description=("A linear probe on a 23M-parameter embedding model, trained on BANKING77. "
                 "Code: https://github.com/shrinidhisuresha/typed-decisions"),
    examples=[["My card hasn't arrived yet"], ["How do I top up with Apple Pay?"],
              ["Why was I charged a fee for withdrawing cash?"]],
)

if __name__ == "__main__":
    demo.launch()
