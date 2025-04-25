from typing import Tuple

import einops
import torch as t
import torch.nn as nn
import tqdm  # type: ignore
import transformer_lens as tl
from jaxtyping import Float, Int
from pydantic import BaseModel
from rich import print
from torch import Tensor
from transformer_lens.utils import gelu_new


class Config(BaseModel):
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12
    device: str = "cuda"


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(
            self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch
        )


class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer(
            "IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=cfg.device)
        )

    def forward(
        self, normalized_resid_pre: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                normalized_resid_pre,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                normalized_resid_pre,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )
        v = (
            einops.einsum(
                normalized_resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        # Take weighted sum of value vectors, according to attention probabilities
        z = einops.einsum(
            v,
            attn_pattern,
            "batch posn_K nheads d_head, batch nheads posn_Q posn_K -> batch posn_Q nheads d_head",
        )

        # Calculate output (by applying matrix W_O and summing over heads, then adding bias b_O)
        attn_out = (
            einops.einsum(
                z,
                self.W_O,
                "batch posn_Q nheads d_head, nheads d_head d_model -> batch posn_Q d_model",
            )
            + self.b_O
        )

        return attn_out

    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        # Define a mask that is True for all positions we want to set probabilities to zero for
        all_ones = t.ones(
            attn_scores.size(-2), attn_scores.size(-1), device=attn_scores.device
        )
        mask = t.triu(all_ones, diagonal=1).bool()
        # Apply the mask to attention scores, then return the masked scores
        attn_scores.masked_fill_(mask, self.IGNORE)
        return attn_scores


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(t.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(t.empty((cfg.d_mlp, cfg.d_model)))
        self.b_in = nn.Parameter(t.zeros((cfg.d_mlp)))
        self.b_out = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, normalized_resid_mid: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        pre = (
            einops.einsum(
                normalized_resid_mid,
                self.W_in,
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
            )
            + self.b_in
        )
        post = gelu_new(pre)
        # post = t.nn.functional.relu(pre)
        mlp_out = (
            einops.einsum(
                post,
                self.W_out,
                "batch position d_mlp, d_mlp d_model -> batch position d_model",
            )
            + self.b_out
        )
        return mlp_out


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(
            cfg.d_model, eps=cfg.layer_norm_eps, elementwise_affine=True
        )
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(
            cfg.d_model, eps=cfg.layer_norm_eps, elementwise_affine=True
        )
        self.mlp = MLP(cfg)

    def forward(
        self, resid_pre: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_model"]:
        resid_mid = self.attn(self.ln1(resid_pre)) + resid_pre
        resid_post = self.mlp(self.ln2(resid_mid)) + resid_mid
        return resid_post


class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        self.b_U = nn.Parameter(t.zeros((cfg.d_vocab), requires_grad=False))

    def forward(
        self, normalized_resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return (
            einops.einsum(
                normalized_resid_final,
                self.W_U,
                "batch posn d_model, d_model d_vocab -> batch posn d_vocab",
            )
            + self.b_U
        )


class DemoTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = t.nn.LayerNorm(
            cfg.d_model, eps=cfg.layer_norm_eps, elementwise_affine=True
        )
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens) + self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual)
        logits = self.unembed(self.ln_final(residual))
        return logits


def from_pretrained(
    model_name: str, device: str = "cuda"
) -> Tuple[DemoTransformer, tl.HookedTransformer]:
    reference_model = tl.HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
    )
    config = Config(
        d_model=reference_model.cfg.d_model,
        d_vocab=reference_model.cfg.d_vocab,
        n_ctx=reference_model.cfg.n_ctx,
        d_head=reference_model.cfg.d_head,
        d_mlp=reference_model.cfg.d_mlp,
        n_heads=reference_model.cfg.n_heads,
        n_layers=reference_model.cfg.n_layers,
    )
    print(
        f"{config.model_dump_json(indent=4)}\n\nParams: {reference_model.cfg.n_params:,}"
    )

    def convert_state_dict_for_torch_layernorm(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if ".ln" in key or "ln_final" in key:
                if key.endswith(".w"):
                    new_key = key.replace(".w", ".weight")
                    new_state_dict[new_key] = value
                elif key.endswith(".b"):
                    new_key = key.replace(".b", ".bias")
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            else:
                new_state_dict[key] = value
        return new_state_dict

    demo_model = DemoTransformer(config).to(device)

    converted_state_dict = convert_state_dict_for_torch_layernorm(
        reference_model.state_dict()
    )
    demo_model.load_state_dict(converted_state_dict, strict=False)
    return demo_model, reference_model


def generate(
    demo_model: DemoTransformer,
    reference_model: tl.HookedTransformer,
    prompt: str,
    max_generated_tokens: int = 32,
) -> Tuple[str, str]:
    if reference_model.tokenizer is None:
        raise ValueError("Tokenizer can not be none")

    demo_completion = prompt
    reference_completion = prompt

    for _ in tqdm.tqdm(range(max_generated_tokens)):
        demo_tokens = reference_model.to_tokens(demo_completion).to(
            demo_model.cfg.device
        )
        reference_tokens = reference_model.to_tokens(reference_completion).to(
            reference_model.cfg.device
        )

        demo_logits = demo_model(demo_tokens)
        reference_logits = reference_model(reference_tokens)

        demo_completion += reference_model.tokenizer.decode(
            demo_logits[-1, -1].argmax()
        )
        reference_completion += reference_model.tokenizer.decode(
            reference_logits[-1, -1].argmax()
        )

    print(f"""
        [blue][b]Demo transformer completion:[/] {demo_completion}[/]
        [red][b]Reference transformer completion:[/] {reference_completion}[/]
    """)

    return demo_completion, reference_completion
