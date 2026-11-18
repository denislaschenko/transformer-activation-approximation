"""Pythia-1B inference experiment: baseline vs activation/softmax replacement.

Designed for the repository layout:
    config.py
    approximations/activations.py  (custom_activation(x))
    approximations/softmax.py      (custom_softmax(x, dim))
    models/pythia.py              (this file)

Run from the repository root with:
    python -m models.pythia

This is an inference-only prototype, tested against the Transformers 5.19.0
GPT-NeoX eager attention source, not against your local environment.
"""

import math

import torch
from torch import nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    GPTNeoXForCausalLM,
    AttentionInterface,
    AttentionMaskInterface,
)
from transformers.masking_utils import eager_mask

from config import Approximation
from approximations.activations import custom_activation
from approximations.softmax import custom_softmax


CHECKPOINT = "EleutherAI/pythia-1b"
ATTENTION_NAME = "pythia_softmax_approx"


class CustomActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_activation(x)


class CustomSoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return custom_softmax(x, dim=self.dim)


SOFTMAX = CustomSoftmax(dim=-1)


def custom_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GPT-NeoX eager attention, with only softmax replaced.

    Reference: transformers v5.19.0, modeling_gpt_neox.eager_attention_forward.
    """
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    # The registered eager_mask creates the causal mask (and any padding mask).
    if attention_mask is None:
        raise RuntimeError(
            "Missing causal attention mask! Check AttentionMaskInterface registration."
        )
    attn_weights = attn_weights + attention_mask

    # GPT-NeoX's exact eager backend explicitly evaluates softmax in float32.
    # Preserve this behavior for the initial equivalence/control experiment.
    attn_weights = SOFTMAX(attn_weights.to(torch.float32)).to(query.dtype)

    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


def register_custom_attention() -> None:
    """Register both attention calculation and its causal-mask formatter."""
    AttentionInterface.register(ATTENTION_NAME, custom_attention_forward)
    AttentionMaskInterface.register(ATTENTION_NAME, eager_mask)


def load_test_tokens(tokenizer, max_tokens: int | None) -> torch.Tensor:
    """Tokenize WikiText-2 RAW test split into one continuous token sequence."""
    test = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="test"
    )
    encoded = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    input_ids = encoded["input_ids"]
    if max_tokens is not None:
        input_ids = input_ids[:, :max_tokens]
    if input_ids.shape[1] < 2:
        raise ValueError("At least two tokens are required for perplexity evaluation.")
    return input_ids


def evaluate_perplexity(
    model: GPTNeoXForCausalLM,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    context_length: int,
    stride: int,
) -> dict[str, float | int]:
    """Strided/sliding-window perplexity with properly masked overlap tokens."""
    if not (1 <= stride <= context_length):
        raise ValueError("Require 1 <= stride <= context_length.")

    model.eval()
    seq_len = input_ids.shape[1]
    prev_end = 0
    nll_sum = 0.0
    token_count = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + context_length, seq_len)
        new_tokens = end - prev_end
        batch = input_ids[:, begin:end].to(device)
        labels = batch.clone()
        labels[:, :-new_tokens] = -100  # overlapping context is not scored again

        # Causal LM internally shifts labels left by one token. Count ONLY valid
        # shifted targets, rather than subtracting one on every window.
        scored_tokens = int((labels[:, 1:] != -100).sum().item())
        if scored_tokens > 0:
            with torch.inference_mode():
                output = model(input_ids=batch, labels=labels, use_cache=False)
            nll_sum += float(output.loss.item()) * scored_tokens
            token_count += scored_tokens

        prev_end = end
        if end == seq_len:
            break

    if token_count == 0:
        raise ValueError("No tokens were scored. Increase context_length.")

    avg_nll = nll_sum / token_count
    return {
        "perplexity": math.exp(avg_nll),
        "mean_nll": avg_nll,
        "evaluated_tokens": token_count,
    }


def run_experiment(
    approximation: Approximation,
    *,
    max_tokens: int | None = 4096,
    context_length: int = 512,
    stride: int = 256,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = GPTNeoXForCausalLM.from_pretrained(
        CHECKPOINT,
        attn_implementation="eager",
        dtype=torch.float32,  # stable exact-control comparison
    ).to(device)
    model.eval()

    context_length = min(context_length, model.config.max_position_embeddings)
    if stride > context_length:
        raise ValueError("stride exceeds the effective context length")

    tokens = load_test_tokens(tokenizer, max_tokens)
    # Small fixed diagnostic sequence for comparing final token logits.
    diagnostic_ids = tokens[:, :min(64, tokens.shape[1])].to(device)
    with torch.inference_mode():
        baseline_diagnostic_logits = model(
            input_ids=diagnostic_ids, use_cache=False
        ).logits.cpu()

    baseline = evaluate_perplexity(
        model, tokens, device=device, context_length=context_length, stride=stride
    )

    if approximation in (Approximation.ACTIVATION, Approximation.BOTH):
        for layer in model.gpt_neox.layers:
            layer.mlp.act = CustomActivation()

    if approximation in (Approximation.ATTENTION, Approximation.BOTH):
        register_custom_attention()
        model.set_attn_implementation(ATTENTION_NAME)

    with torch.inference_mode():
        modified_diagnostic_logits = model(
            input_ids=diagnostic_ids, use_cache=False
        ).logits.cpu()
    logit_error = (baseline_diagnostic_logits - modified_diagnostic_logits).abs()

    modified = evaluate_perplexity(
        model, tokens, device=device, context_length=context_length, stride=stride
    )

    return {
        "checkpoint": CHECKPOINT,
        "dataset": "Salesforce/wikitext: wikitext-2-raw-v1/test",
        "approximation": str(approximation),
        "device": str(device),
        "context_length": context_length,
        "stride": stride,
        "input_tokens": int(tokens.numel()),
        "baseline": baseline,
        "modified": modified,
        "delta_perplexity": modified["perplexity"] - baseline["perplexity"],
        "delta_mean_nll": modified["mean_nll"] - baseline["mean_nll"],
        "diagnostic_logit_mae": logit_error.mean().item(),
        "diagnostic_logit_max_abs_error": logit_error.max().item(),
        "diagnostic_prediction_disagreements": int(
            (baseline_diagnostic_logits.argmax(-1) != modified_diagnostic_logits.argmax(-1))
            .sum().item()
        ),
    }


if __name__ == "__main__":
    import pprint
    pprint.pp(run_experiment(Approximation.BOTH))
