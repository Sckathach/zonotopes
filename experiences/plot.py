import circuitsvis as cv
import transformer_lens as tl
from IPython.display import display


def plot_attn_patterns(
    model: tl.HookedTransformer, cache: tl.ActivationCache, prompt: str
):
    str_tokens = model.to_str_tokens(prompt)
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(
            cv.attention.attention_patterns(
                tokens=str_tokens, attention=attention_pattern
            )
        )
