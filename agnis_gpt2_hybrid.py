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
        Path("/kaggle/working"),
        Path.cwd(),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        search_roots.append(input_root)
        for sub in input_root.iterdir():
            if sub.is_dir():
                if 'fineweb' in sub.name.lower() or 'chunk' in sub.name.lower():
                    continue
                search_roots.append(sub)

    patterns = [
        "agnis_v5_30m_fluency.pt",
        "agnis_v5_sprint4*.pt",
        "agnis_sprint4_best*.pt",
        "agnis_v5*.pt",
    ]
    matches: list[Path] = []
    
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(list(root.glob(pattern)))
            matches.extend(list(root.glob(f"*/{pattern}")))
            matches.extend(list(root.glob(f"*/*/{pattern}")))
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
        
        self.deep_layers = [0, 3, 6, 9]
        self.gamma_max = 0.20
        self._current_agnis_h = None
        self.gate_stats = {l: 0.0 for l in self.deep_layers}
        
        self.is_replay = False
        
        self.stored_logits = {l: [] for l in self.deep_layers}
        self.store_gate_logits = False
        
        self.deep_projs = nn.ModuleDict({
            str(l): nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim)
            ) for l in self.deep_layers
        }).to(self.device)
        
        self.deep_gates = nn.ModuleDict({
            str(l): nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for l in self.deep_layers
        }).to(self.device)

        self._init_deep_modules()
        self._register_deep_hooks()
        self._load_agnis_core(self.agnis_checkpoint_path)
        self.freeze_agnis()
        self.freeze_gpt2()

    def _init_deep_modules(self) -> None:
        for name, m in self.deep_projs.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)
        
        # Initialize 2-layer MLP gates
        for l in self.deep_layers:
            gate_mlp = self.deep_gates[str(l)]
            nn.init.xavier_uniform_(gate_mlp[1].weight, gain=0.1)
            nn.init.zeros_(gate_mlp[1].bias)
            
            nn.init.normal_(gate_mlp[3].weight, std=1e-3)
            # Default gate output to negative (silent start)
            nn.init.constant_(gate_mlp[3].bias, -2.5)
                
        # Initialize final linear in deep_projs very small
        for l in self.deep_layers:
            final_linear = self.deep_projs[str(l)][3]
            nn.init.normal_(final_linear.weight, std=1e-3)
            nn.init.zeros_(final_linear.bias)

    def _register_deep_hooks(self) -> None:
        self._hooks = []
        for l in self.deep_layers:
            def make_hook(layer_idx):
                def hook(module, inputs):
                    hidden_states = inputs[0]
                    if self._current_agnis_h is not None:
                        seq_len = hidden_states.shape[1]
                        agnis_h_t = self._current_agnis_h[:, -seq_len:, :]
                        
                        proj = self.deep_projs[str(layer_idx)](agnis_h_t)
                        gate_logits = self.deep_gates[str(layer_idx)](agnis_h_t)
                        
                        if getattr(self, "store_gate_logits", False):
                            self.stored_logits[layer_idx].append(gate_logits)
                        
                        # V3.7: Detach gamma_l to decouple gate parameter learning from LM gradients
                        # We apply hard thresholding only during evaluation to prevent blocking gradients during training.
                        raw_gamma = torch.sigmoid(gate_logits) * self.gamma_max
                        if self.deep_gates.training:
                            gamma_l = raw_gamma.detach()
                        else:
                            gamma_l = torch.where(raw_gamma < 0.1 * self.gamma_max, torch.zeros_like(raw_gamma), raw_gamma).detach()
                        self.gate_stats[layer_idx] = gamma_l.mean().item()
                        
                        # V3.6: Norm-calibrated injection
                        h_norm = hidden_states.norm(dim=-1, keepdim=True).detach()
                        p_norm = proj.norm(dim=-1, keepdim=True)
                        proj_calibrated = (proj / (p_norm + 1e-8)) * h_norm
                        
                        hidden_states = hidden_states + gamma_l * proj_calibrated
                        
                    return (hidden_states,) + inputs[1:]
                return hook
            
            handle = self.gpt2.transformer.h[l].register_forward_pre_hook(make_hook(l))
            self._hooks.append(handle)

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
        self._current_agnis_h = agnis_hidden
        return agnis_hidden, None, source_embeds

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        _, _, fused_embeds = self.build_gpt2_inputs(input_ids)
        outputs = self.gpt2(inputs_embeds=fused_embeds, labels=labels)
        outputs.adapted_embeds = None
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
            params = [
                {"params": self.deep_projs.parameters(), "lr": 1e-3},
                {"params": self.deep_gates.parameters(), "lr": 1e-3}
            ]
        elif phase == 2:
            self.freeze_agnis()
            self.unfreeze_gpt2_last_layers(num_layers=2)
            params = [
                {"params": list(self.deep_projs.parameters()) + list(self.deep_gates.parameters()), "lr": 1e-3},
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
            "deep_projs": self.deep_projs.state_dict(),
            "deep_gates": self.deep_gates.state_dict(),
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
