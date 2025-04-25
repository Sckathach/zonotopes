from typing import Literal, Tuple

import circuitsvis as cv
import einops
import torch as t
import torch.nn as nn
import transformer_lens as tl
from huggingface_hub import hf_hub_download
from IPython.display import display
from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import Tensor


class Config(BaseModel):
    init_range: float = 0.02
    d_model: int = 768
    d_head: int = 64
    n_heads: int = 12
    n_layers: int = 2
    n_ctx: int = 2048
    d_vocab: int = 50278
    device: str = "cuda"


class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        return self.W_E[tokens]


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch seq_len"]
    ) -> Float[Tensor, "batch seq_len d_model"]:
        batch, seq_len = tokens.shape
        return einops.repeat(
            self.W_pos[:seq_len],
            "seq_len d_model -> batch seq_len d_model",
            batch=batch,
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
        self.mask = nn.Parameter(t.ones((cfg.n_ctx, cfg.n_ctx)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer(
            "IGNORE", t.tensor(float("-inf"), dtype=t.float32, device=cfg.device)
        )

    def forward(
        self,
        resid_pre: Float[Tensor, "batch posn d_model"],
        shortformer_pos_embed: Float[Tensor, "batch posn d_model"],
    ) -> Float[Tensor, "batch posn d_model"]:
        attn_pattern = self.compute_pattern(resid_pre, shortformer_pos_embed)

        v = (
            einops.einsum(
                resid_pre,
                self.W_V,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_V
        )

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

    def compute_pattern(
        self,
        resid_pre: Float[Tensor, "batch posn d_model"],
        shortformer_pos_embed: Float[Tensor, "batch posn d_model"],
    ) -> Float[Tensor, "batch posn d_model"]:
        # Calculate query, key and value vectors
        q = (
            einops.einsum(
                resid_pre + shortformer_pos_embed,
                self.W_Q,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_Q
        )
        k = (
            einops.einsum(
                resid_pre + shortformer_pos_embed,
                self.W_K,
                "batch posn d_model, nheads d_model d_head -> batch posn nheads d_head",
            )
            + self.b_K
        )

        # Calculate attention scores, then scale and mask, and apply softmax to get probabilities
        attn_scores = einops.einsum(
            q,
            k,
            "batch posn_Q nheads d_head, batch posn_K nheads d_head -> batch nheads posn_Q posn_K",
        )
        attn_scores_masked = self.apply_causal_mask(attn_scores / self.cfg.d_head**0.5)
        attn_pattern = attn_scores_masked.softmax(-1)

        return attn_pattern

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


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.attn = Attention(cfg)

    def forward(
        self,
        resid_pre: Float[Tensor, "batch position d_model"],
        shortformer_pos_embed: Float[Tensor, "batch position d_model"],
    ) -> Float[Tensor, "batch position d_model"]:
        return self.attn(resid_pre, shortformer_pos_embed) + resid_pre


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


class AttnOnlyTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.unembed = Unembed(cfg)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        residual = self.embed(tokens)
        shortformer_pos_embed = self.pos_embed(tokens)
        for block in self.blocks:
            residual = block(residual, shortformer_pos_embed)
        logits = self.unembed(residual)
        return logits


def load_model(
    model_name: Literal["attn_only_2L_half"] = "attn_only_2L_half",
) -> Tuple[AttnOnlyTransformer, tl.HookedTransformer]:
    match model_name:
        case "attn_only_2L_half":
            cfg = tl.HookedTransformerConfig(
                d_model=768,
                d_head=64,
                n_heads=12,
                n_layers=2,
                n_ctx=2048,
                d_vocab=50278,
                attention_dir="causal",
                attn_only=True,  # defaults to False
                tokenizer_name="EleutherAI/gpt-neox-20b",
                seed=398,
                use_attn_result=True,
                normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
                positional_embedding_type="shortformer",
            )

            repo_id = "callummcdougall/attn_only_2L_half"
            filename = "attn_only_2L_half.pth"
            device = "cuda"

            weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
            model = tl.HookedTransformer(cfg)
            pretrained_weights = t.load(
                weights_path, map_location=device, weights_only=True
            )
            model.load_state_dict(pretrained_weights)

            attn_model = AttnOnlyTransformer(
                Config(
                    d_model=768,
                    d_head=64,
                    n_heads=12,
                    n_layers=2,
                    n_ctx=2048,
                    d_vocab=50278,
                )
            ).to(model.cfg.device)
            attn_model.load_state_dict(pretrained_weights)

            return attn_model, model

        case _:
            raise ValueError("This model is not supported yet.")


def plot_attn_patterns(
    model: tl.HookedTransformer, cache: tl.ActivationCache, prompt: str
):
    str_tokens = model.to_str_tokens(prompt)
    for layer in range(model.cfg.n_layers):
        attention_pattern = cache["pattern", layer]
        display(
            cv.attention.attention_patterns(  # type: ignore
                tokens=str_tokens, attention=attention_pattern
            )
        )
