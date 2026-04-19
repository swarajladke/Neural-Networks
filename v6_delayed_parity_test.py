import random

import torch

from agnis_v4_core import PredictiveHierarchy


def build_delayed_parity():
    samples = []
    for i in range(16):
        bits = torch.tensor([float(b) for b in format(i, "04b")], dtype=torch.float32).unsqueeze(0)
        target = torch.tensor([[float(bits.sum().item() % 2)]], dtype=torch.float32)
        blank = torch.zeros_like(bits)
        samples.append((bits, blank, target))
    return samples


def run_sequence(
    hierarchy: PredictiveHierarchy,
    x_bits: torch.Tensor,
    blank: torch.Tensor,
    target: torch.Tensor | None = None,
    update: bool = False,
):
    hierarchy.reset_states(batch_size=1)
    hierarchy.predict_label(x_bits, max_steps=15, update_temporal=True)
    hierarchy.predict_label(blank, max_steps=10, update_temporal=True)

    if update:
        if target is None:
            raise ValueError("target is required when update=True")
        hierarchy.infer_and_learn(
            blank,
            top_level_label=target,
            max_steps=25,
            recognition_weight=1.0,
            beta_push=6.0,
            warm_start=True,
        )
        return None

    return hierarchy.predict_label(blank, max_steps=20, update_temporal=False)


def evaluate(hierarchy: PredictiveHierarchy, samples):
    correct = 0
    with torch.no_grad():
        for x_bits, blank, target in samples:
            pred = run_sequence(hierarchy, x_bits, blank, update=False)
            pred_bit = float(torch.sigmoid(pred[:, :1]).item() > 0.5)
            if pred_bit == float(target.item()):
                correct += 1
    return correct / len(samples)


def train_delayed_parity():
    print("==================================================")
    print(" V6.1 RECURRENCE CHALLENGE: DELAYED PARITY")
    print("==================================================")

    torch.manual_seed(42)
    random.seed(42)

    samples = build_delayed_parity()
    hierarchy = PredictiveHierarchy([4, 16, 8, 1], device="cpu")

    epochs = 10
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        random.shuffle(samples)
        for x_bits, blank, target in samples:
            run_sequence(hierarchy, x_bits, blank, target=target, update=True)

        acc = evaluate(hierarchy, samples)
        best_acc = max(best_acc, acc)
        if epoch % 5 == 0 or acc == 1.0:
            print(f"Epoch {epoch:02d} | Accuracy: {acc:.3f}")
        if acc == 1.0:
            break

    print(f"\nBest delayed-parity accuracy: {best_acc:.3f}")
    if best_acc >= 0.9:
        print("[PASS] Recurrent state preserved the parity cue across the delay.")
    else:
        print("[INFO] Accuracy stayed below the target threshold; treat this as an experimental challenge, not a release gate.")


if __name__ == "__main__":
    train_delayed_parity()
