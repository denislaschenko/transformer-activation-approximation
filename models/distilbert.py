import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AttentionInterface, AttentionMaskInterface
from transformers.processing_utils import Unpack
from transformers.masking_utils import eager_mask
from transformers.utils.generic import TransformersKwargs

from config import Approximation
from approximations.activations import custom_activation
from approximations.softmax import custom_softmax

def run_experiment(approximation: Approximation):
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, attn_implementation="eager")

    data = pd.read_parquet("./data/sst-2/validation.parquet")

    test_data = data["sentence"].tolist()
    labels = data["label"].tolist()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    baseline_logits = evaluate_model(model, tokenizer, test_data)

    if approximation in (Approximation.ACTIVATION, Approximation.BOTH):
        for block in model.distilbert.transformer.layer:
            block.ffn.activation = CustomActivation()

    AttentionInterface.register("attention_approx", custom_attention_forward)
    AttentionMaskInterface.register("attention_approx", eager_mask)

    if approximation in (Approximation.ATTENTION, Approximation.BOTH):
        model.set_attn_implementation("attention_approx")

    modified_logits = evaluate_model(model, tokenizer, test_data)

    baseline_classes = baseline_logits.argmax(dim=-1)
    modified_classes = modified_logits.argmax(dim=-1)

    baseline_correct = 0
    modified_correct = 0

    for index, label in enumerate(labels):
        if labels[index] == baseline_classes[index]:
            baseline_correct += 1

        if labels[index] == modified_classes[index]:
            modified_correct += 1

    baseline_accuracy = baseline_correct/len(labels)
    modified_accuracy = modified_correct/len(labels)

    return {
    "baseline_accuracy": baseline_accuracy,
    "modified_accuracy": modified_accuracy,
    "accuracy_difference": modified_accuracy - baseline_accuracy,
    "max_logit_difference": (
        baseline_logits - modified_logits
    ).abs().max().item(),
    "mean_logit_difference": (
        baseline_logits - modified_logits
    ).abs().mean().item(),
    "prediction_disagreements": (
        baseline_classes != modified_classes
    ).sum().item()
}


class CustomActivation(nn.Module):
    def forward(self, x):
        return custom_activation(x)

class CustomSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return custom_softmax(x, self.dim)

softmax = CustomSoftmax(dim=-1)

def evaluate_model(model, tokenizer, texts, batch_size=32):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        all_logits = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]

            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)

            with torch.inference_mode():
                output = model(**tokens)

            all_logits.append(output.logits.cpu())

        return torch.cat(all_logits)

def custom_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if scaling is None:
            scaling = float(query.size(-1) ** -0.5)

        attn_weights = torch.matmul(query, key.transpose(2,3)) * scaling

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = softmax(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1,2).contiguous()

        return attn_output, attn_weights