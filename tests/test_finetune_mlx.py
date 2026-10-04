import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from typed_decisions.backend_impls.mlx import _Tokenizer
from typed_decisions.benchmarks import Sample
from typed_decisions.finetune.data import _score, build, encode
from typed_decisions.finetune.train_mlx import LORA_KEYS, answer_logits, collate, loss_fn

TOKENIZER = "hf-internal-testing/tiny-random-LlamaForCausalLM"
LABELS = [f"intent {i}" for i in range(6)]
SAMPLES = [Sample(f"message number {i}", LABELS[i % 6]) for i in range(24)]


def tiny(tied: bool = True):
    """A tiny random Llama in MLX: checks the training plumbing, not model quality."""
    from mlx_lm.models import llama
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    args = llama.ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2,
                           intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-5,
                           vocab_size=len(tok), num_key_value_heads=2,
                           tie_word_embeddings=tied)
    mx.random.seed(0)
    model = llama.Model(args)
    mx.eval(model.parameters())
    return model, tok


def encoded(tok, n=4):
    rows = build(SAMPLES, LABELS, null_share=0.0, noul_share=0.0)[:n]
    return [encode(r, _Tokenizer(tok)) for r in rows]


@pytest.mark.parametrize("tied", [True, False])
def test_answer_logits_match_the_full_forward_at_the_answer_position(tied):
    model, tok = tiny(tied)
    batch = encoded(tok)
    ids, last, *_ = collate(batch, tok.eos_token_id)
    full = model(ids)[mx.arange(ids.shape[0]), last]
    assert mx.allclose(answer_logits(model, ids, last), full, atol=1e-4)


def test_right_padding_does_not_reach_a_rows_answer_logits():
    # Same shapes, different pad token: causal attention means the padded row's answer
    # position never sees the pads, so the logits must be bit-identical. (Comparing against
    # an unpadded batch differs by ~1e-3 from matmul shapes alone, which would hide a leak.)
    model, tok = tiny()
    batch = sorted(encoded(tok), key=lambda e: len(e.input_ids))
    pair = [batch[0], batch[-1]]
    assert tok.eos_token_id != 0
    assert len(pair[0].input_ids) < len(pair[1].input_ids)
    a = answer_logits(model, *collate(pair, tok.eos_token_id)[:2])
    b = answer_logits(model, *collate(pair, 0)[:2])
    assert mx.array_equal(a, b)


def test_score_rows_add_rps_and_the_loss_is_finite():
    import random
    model, tok = tiny()
    row = _score("a fairly urgent message", "mid", ["low", "mid", "high"],
                 random.Random(0), ("How urgent?",))
    e = encode(row, _Tokenizer(tok))
    args = collate([e], tok.eos_token_id)
    plain = loss_fn(model, *args, [e.ordinal], 0.0)
    with_rps = loss_fn(model, *args, [e.ordinal], 1.0)
    assert mx.isfinite(plain).item() and with_rps.item() > plain.item()


def test_lora_steps_lower_the_loss_and_train_only_the_adapter():
    from mlx_lm.tuner.utils import linear_to_lora_layers
    model, tok = tiny()
    model.freeze()
    linear_to_lora_layers(model, len(model.layers),
                          {"rank": 4, "scale": 2.0, "dropout": 0.0, "keys": LORA_KEYS})
    trainable = dict(tree_flatten(model.trainable_parameters()))
    assert trainable and all("lora" in k for k in trainable)

    batch = encoded(tok)
    args = collate(batch, tok.eos_token_id)
    ordinal = [e.ordinal for e in batch]
    step = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=1e-2)
    first, _ = step(model, *args, ordinal, 0.0)
    for _ in range(20):
        loss, grads = step(model, *args, ordinal, 0.0)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state)
    assert loss.item() < first.item() * 0.5
