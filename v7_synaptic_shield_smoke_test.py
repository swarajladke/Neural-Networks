import torch

from agnis_v4_core import PredictiveHierarchy


def run_synaptic_shield_smoke_test():
    print("==================================================")
    print(" V7.3: SYNAPTIC SHIELD SMOKE TEST")
    print("==================================================")

    torch.manual_seed(7)

    hierarchy = PredictiveHierarchy([8, 8, 8], device="cpu")
    x = torch.randn(2, 8)

    hierarchy.reset_states(batch_size=2)
    baseline = hierarchy.infer_with_manifold_slice(x, slice_end=8, max_steps=30).detach().clone()

    hierarchy.force_recruit_language_sliver(n=4, language="russian")
    hierarchy.set_experts_bias(8, 12, -10.0)

    hierarchy.reset_states(batch_size=2)
    isolated = hierarchy.infer_with_manifold_slice(x, slice_end=8, max_steps=30).detach().clone()

    drift = torch.mean((baseline - isolated) ** 2).item()
    print(f"Baseline vs isolated drift: {drift:.8f}")

    if drift < 1e-8:
        print("[PASS] Base manifold remains stable under isolated inference after expansion.")
    else:
        raise AssertionError(f"Synaptic shield drift too large: {drift:.8f}")


if __name__ == "__main__":
    run_synaptic_shield_smoke_test()
