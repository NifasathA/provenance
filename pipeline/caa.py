"""Contrastive Activation Addition (CAA) primitives.

Pure functions: no I/O, no globals. The orchestrator (train_vectors.py)
handles the corpus structure and caching; this module only knows about
activations and vectors.
"""

import torch


def get_last_token_activation(model, tokenizer, chat_messages, layer):
    """Run a chat-formatted forward pass and return the residual-stream
    activation at the layer's output, at the last token of the sequence
    (i.e. the end of the final assistant message).

    Returns a float32 CPU tensor of shape [hidden_dim].
    """
    text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states[layer + 1]  # [1, seq, hidden]; +1 because index 0 is embeddings
    return hs[0, -1].float().cpu()


def caa_vector(positive_activations, negative_activations):
    """Compute a unit-normed CAA direction from two sets of activations.

    vec = mean(positive) - mean(negative), normalized to unit length.
    """
    pos = torch.stack(positive_activations).mean(dim=0)
    neg = torch.stack(negative_activations).mean(dim=0)
    vec = pos - neg
    return vec / vec.norm()


def orthogonalize(vec, against):
    """Remove the component of `vec` that lies along `against`.

    Used to drive a vector below the similarity threshold against a
    specific other vector when the raw CAA result is too correlated.
    Returns a re-normalized direction.
    """
    against = against / against.norm()
    projection = torch.dot(vec, against) * against
    out = vec - projection
    return out / out.norm()
