from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from agnis_v4_core import PredictiveHierarchy


DEFAULT_MODEL_NAME = "gpt2"
DEFAULT_EMBED_DIM = 768
DEFAULT_HIDDEN_DIM = 3072
DEFAULT_MAX_SETTLE_STEPS = 5
DEFAULT_SAVE_NAME = "agnis_gpt2_hybrid.pt"
DEFAULT_PROMPTS = [
    "The history of artificial intelligence",
    "Once upon a time in a land far away",
    "The scientist discovered that",
    "In the year 2050, humans will",
]


@dataclass
class TrainPhaseConfig:
    steps: int
    adapter_lr: float
    gpt2_lr: float = 0.0
    save_every: int = 2000
    log_every: int = 100
    generation_every: int = 1000
    batch_size: int = 4
    seq_len: int = 128


def resolve_save_dir() -> Path:
    kaggle_dir = Path("/kaggle/working")
    if kaggle_dir.exists():
        return kaggle_dir
    return Path.cwd()


def find_agnis_checkpoint(explicit_path: str | os.PathLike[str] | None = None) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"AGNIS checkpoint not found: {candidate}")

    search_roots = [
        Path("/kaggle/input/agnis-ckpt"),
        Path("/kaggle/working"),
        Path.cwd(),
    ]
    patterns = [
        "agnis_v5_sprint4*.pt",
        "agnis_sprint4_best*.pt",
        "agnis_v5_30m_fluency.pt",
        "agnis_v5*.pt",
    ]
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError("No AGNIS V5 checkpoint found in Kaggle input or working directory.")
    matches.sort(key=lambda path: path.stat().st_size, reverse=True)
    return matches[0]


def stream_fineweb_edu(tokenizer: GPT2Tokenizer, seq_len: int, batch_size: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    from datasets import load_dataset

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    buffer: list[int] = []
    chunk_len = seq_len + 1
    batch_tokens = chunk_len * batch_size

    for row in dataset:
        text = row.get("text", "").strip()
        if len(text) < 32:
            continue
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < chunk_len:
            continue
        buffer.extend(token_ids)

        while len(buffer) >= batch_tokens:
            current = buffer[:batch_tokens]
            del buffer[:batch_tokens]
            batch = torch.tensor(current, dtype=torch.long).view(batch_size, chunk_len)
            yield batch[:, :-1], batch[:, 1:]


class AgnisGpt2Hybrid(nn.Module):
    def __init__(
        self,
        agnis_checkpoint: str | os.PathLike[str] | None = None,
        model_name: str = DEFAULT_MODEL_NAME,
        embed_dim: int = DEFAULT_EMBED_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        max_settle_steps: int = DEFAULT_MAX_SETTLE_STEPS,
        device: str | None = None,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_settle_steps = max_settle_steps
        self.model_name = model_name
        self.save_dir = resolve_save_dir()
        self.agnis_checkpoint_path = find_agnis_checkpoint(agnis_checkpoint)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name, local_files_only=local_files_only).to(self.device)
        self.gpt2.config.pad_token_id = self.tokenizer.pad_token_id

        self.agnis_core = PredictiveHierarchy([embed_dim, hidden_dim, embed_dim], device=str(self.device)).to(self.device)
        self.adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        ).to(self.device)

        self._init_adapter()
        self._load_agnis_core(self.agnis_checkpoint_path)
        self.freeze_agnis()
        self.freeze_gpt2()

    def _init_adapter(self) -> None:
        # Xavier uniform with small gain so adapter starts near-zero but NOT dead.
        # Zero-init caused flat loss=7.7 for 20k steps (adapter output=0 always).
        for m in self.adapter:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        # LayerNorm: identity init
        nn.init.ones_(self.adapter[3].weight)
        nn.init.zeros_(self.adapter[3].bias)

    def _load_agnis_core(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError(f"Unsupported checkpoint format in {checkpoint_path}")

        skip_suffixes = {
            ".x",
            ".x_temporal",
            ".x_temporal_2",
            ".x_temporal_3",
        }
        hierarchy_state = {
            key[len("hierarchy."):]: value
            for key, value in state_dict.items()
            if key.startswith("hierarchy.")
            and not any(key.endswith(suffix) for suffix in skip_suffixes)
        }
        if not hierarchy_state:
            raise ValueError(f"No hierarchy weights found in {checkpoint_path}")

        missing, unexpected = self.agnis_core.load_state_dict(hierarchy_state, strict=False)
        if unexpected:
            raise ValueError(f"Unexpected AGNIS hierarchy keys: {unexpected[:5]}")
        if missing:
            ignored_substrings = (
                "W_mask",
                "V_mask",
                ".x",
                ".x_temporal",
                ".x_temporal_2",
                ".x_temporal_3",
            )
            missing = [
                key for key in missing
                if not any(fragment in key for fragment in ignored_substrings)
            ]
            if missing:
                raise ValueError(f"Missing AGNIS hierarchy keys: {missing[:5]}")

    def freeze_agnis(self) -> None:
        self.agnis_core.eval()
        for param in self.agnis_core.parameters():
            param.requires_grad_(False)

    def freeze_gpt2(self) -> None:
        self.gpt2.eval()
        for param in self.gpt2.parameters():
            param.requires_grad_(False)

    def unfreeze_gpt2_last_layers(self, num_layers: int = 2) -> None:
        self.freeze_gpt2()
        for block in self.gpt2.transformer.h[-num_layers:]:
            for param in block.parameters():
                param.requires_grad_(True)
        for param in self.gpt2.transformer.ln_f.parameters():
            param.requires_grad_(True)

    def _token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gpt2.transformer.wte(input_ids)

    @torch.no_grad()
    def compute_agnis_hidden(self, input_ids: torch.Tensor, reset_state: bool = True) -> torch.Tensor:
        token_embeds = self._token_embeddings(input_ids).detach()
        token_embeds = F.normalize(token_embeds, dim=-1)
        batch_size, seq_len, _ = token_embeds.shape

        if reset_state:
            self.agnis_core.reset_states(batch_size=batch_size)

        outputs: list[torch.Tensor] = []
        for step in range(seq_len):
            current = token_embeds[:, step, :]
            agnis_out = self.agnis_core.predict_label(
                current,
                max_steps=self.max_settle_steps,
                update_temporal=True,
            )
            outputs.append(agnis_out[:, : self.embed_dim].detach())

        return torch.stack(outputs, dim=1)

    def build_gpt2_inputs(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_embeds = self._token_embeddings(input_ids)
        agnis_hidden = self.compute_agnis_hidden(input_ids, reset_state=True)
        adapted = self.adapter(agnis_hidden)
        fused_embeds = source_embeds + adapted
        return agnis_hidden, adapted, fused_embeds

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        _, adapted, fused_embeds = self.build_gpt2_inputs(input_ids)
        outputs = self.gpt2(inputs_embeds=fused_embeds, labels=labels)
        outputs.adapted_embeds = adapted
        return outputs

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        self.eval()
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        generated = input_ids.clone()

        for _ in range(max_tokens):
            _, _, fused_embeds = self.build_gpt2_inputs(generated)
            outputs = self.gpt2(inputs_embeds=fused_embeds)
            logits = outputs.logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                top_values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits = logits.masked_fill(logits < top_values[:, [-1]], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def build_phase_optimizer(self, phase: int):
        if phase == 1:
            self.freeze_agnis()
            self.freeze_gpt2()
            params = [{"params": self.adapter.parameters(), "lr": 1e-3}]
        elif phase == 2:
            self.freeze_agnis()
            self.unfreeze_gpt2_last_layers(num_layers=2)
            params = [
                {"params": self.adapter.parameters(), "lr": 1e-3},
                {
                    "params": [param for param in self.gpt2.parameters() if param.requires_grad],
                    "lr": 1e-4,
                },
            ]
        else:
            raise ValueError(f"Unsupported phase: {phase}")
        return torch.optim.AdamW(params, weight_decay=0.01)

    def save_hybrid_checkpoint(
        self,
        path: str | os.PathLike[str] | None = None,
        step: int = 0,
        phase: int = 0,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> Path:
        checkpoint_path = Path(path) if path else self.save_dir / DEFAULT_SAVE_NAME
        state = {
            "step": step,
            "phase": phase,
            "agnis_checkpoint_path": str(self.agnis_checkpoint_path),
            "adapter": self.adapter.state_dict(),
            "gpt2_trainable": {
                key: value.detach().cpu()
                for key, value in self.gpt2.state_dict().items()
                if key.startswith("transformer.h.10.")
                or key.startswith("transformer.h.11.")
                or key.startswith("transformer.ln_f.")
            },
        }
        if optimizer is not None:
            state["optimizer"] = optimizer.state_dict()
        torch.save(state, checkpoint_path)
        return checkpoint_path

    def continual_learn_facts(
        self,
        facts: Sequence[str],
        passes: int = 5,
        beta_push: float = 3.0,
    ) -> None:
        self.agnis_core.train()
        for _ in range(passes):
            for fact in facts:
                encoded = self.tokenizer(fact, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"].to(self.device)
                if input_ids.shape[1] < 2:
                    continue
                embeds = self._token_embeddings(input_ids).detach()
                embeds = F.normalize(embeds, dim=-1)
                self.agnis_core.reset_states(batch_size=1)
                for idx in range(input_ids.shape[1] - 1):
                    x = embeds[:, idx, :]
                    y = embeds[:, idx + 1, :]
                    self.agnis_core.infer_and_learn_online(
                        x,
                        top_level_label=y,
                        max_steps=self.max_settle_steps,
                        warm_start=True,
                        beta_push=beta_push,
                    )
        self.freeze_agnis()

    def continual_learning_test(self) -> list[tuple[str, str]]:
        facts = [
            "The codename of the AGNIS lunar base is Selene Station.",
            "Project Aster uses cobalt glass batteries for long missions.",
            "The Europa research vessel is called Blue Meridian.",
            "Dr. Mira Sol leads the AGNIS exobiology division.",
            "The year 2084 treaty was signed in Reykjavik.",
        ]
        self.continual_learn_facts(facts)
        prompts = [
            "The codename of the AGNIS lunar base is",
            "Dr. Mira Sol leads",
            "The year 2084 treaty was signed in",
        ]
        results = []
        for prompt in prompts:
            results.append((prompt, self.generate(prompt, max_tokens=30)))
        return results

    def run_verification_tests(self) -> dict[str, object]:
        sample_text = "The history of artificial intelligence"
        tokens = self.tokenizer(sample_text, return_tensors="pt")
        input_ids = tokens["input_ids"].to(self.device)

        agnis_hidden = self.compute_agnis_hidden(input_ids)
        gpt2_params = sum(param.numel() for param in self.gpt2.parameters())
        agnis_out, adapted, fused = self.build_gpt2_inputs(input_ids)
        gpt2_out = self.gpt2(inputs_embeds=fused)
        generation = self.generate("The history of", max_tokens=20)

        return {
            "test1_shape": tuple(agnis_hidden.shape),
            "test2_params": gpt2_params,
            "test3_agnis_shape": tuple(agnis_out.shape),
            "test3_adapter_shape": tuple(adapted.shape),
            "test3_logits_shape": tuple(gpt2_out.logits.shape),
            "test4_generation": generation,
        }

    def train_phase(
        self,
        phase: int,
        config: TrainPhaseConfig,
        prompts: Sequence[str] | None = None,
    ) -> None:
        prompts = list(prompts or DEFAULT_PROMPTS)
        optimizer = self.build_phase_optimizer(phase)
        self.train()

        batch_iter = stream_fineweb_edu(
            tokenizer=self.tokenizer,
            seq_len=config.seq_len,
            batch_size=config.batch_size,
        )

        for step, (input_ids, labels) in enumerate(batch_iter, start=1):
            if step > config.steps:
                break

            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad(set_to_none=True)
            outputs = self.forward(input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for param in self.parameters() if param.requires_grad],
                1.0,
            )
            optimizer.step()

            if step % config.log_every == 0 or step == 1:
                print(f"Phase {phase} Step {step}: loss={loss.item():.4f}")

            if step % config.generation_every == 0:
                for prompt in prompts:
                    text = self.generate(prompt, max_tokens=100)
                    print(f"Step {step}: prompt={prompt!r}")
                    print(text)

            if step % config.save_every == 0:
                checkpoint_path = self.save_hybrid_checkpoint(
                    path=self.save_dir / DEFAULT_SAVE_NAME,
                    step=step,
                    phase=phase,
                    optimizer=optimizer,
                )
                print(f"Saved checkpoint: {checkpoint_path}")


def print_verification_report(results: dict[str, object]) -> None:
    print(f"Test 1: AGNIS output shape: {results['test1_shape']}")
    print(f"Test 2: GPT-2 params: {results['test2_params']:,}")
    print(
        "Test 3: "
        f"AGNIS={results['test3_agnis_shape']} | "
        f"Adapter={results['test3_adapter_shape']} | "
        f"Logits={results['test3_logits_shape']}"
    )
    print(f"Test 4: Generation: {results['test4_generation']}")


def main() -> None:
    hybrid = AgnisGpt2Hybrid()
    results = hybrid.run_verification_tests()
    print_verification_report(results)


if __name__ == "__main__":
    main()
