import torch

from agnis_v4_core import PredictiveHierarchy


def run_synaptic_shield_smoke_test():
    print("==================================================")
    print(" V7.3: SYNAPTIC SHIELD SMOKE TEST")
    print("==================================================")

    torch.manual_seed(7)

    hierarchy = PredictiveHierarchy([8, 8, 8], device="cpu")
    x = torch.randn(2, 8)
    base_dim = 8

    V_id_before = id(hierarchy.layers[0].V)
    W_id_before = id(hierarchy.layers[0].W)
    with hierarchy.manifold_gate(0, base_dim):
        V_id_during = id(hierarchy.layers[0].V)
        W_id_during = id(hierarchy.layers[0].W)
        assert V_id_before == V_id_during, "FAIL: V replaced during gate"
        assert W_id_before == W_id_during, "FAIL: W replaced during gate"
    assert V_id_before == id(hierarchy.layers[0].V), "FAIL: V not restored"
    assert W_id_before == id(hierarchy.layers[0].W), "FAIL: W not restored"
    print("PASS: weight objects never replaced")

    hierarchy.reset_states(batch_size=2)
    baseline = hierarchy.infer_with_manifold_slice(x, slice_end=8, max_steps=30).detach().clone()

    hierarchy.force_recruit_language_sliver(n=4, language="russian")
    hierarchy.set_experts_bias(8, 12, -10.0)

    hierarchy.reset_states(batch_size=2)
    isolated = hierarchy.infer_with_manifold_slice(x, slice_end=8, max_steps=30).detach().clone()

    drift = torch.mean((baseline - isolated) ** 2).item()
    print(f"Baseline vs isolated drift: {drift:.8f}")

    if drift < 1e-4:
        print("[PASS] Base manifold remains stable under isolated inference after expansion.")
    else:
        raise AssertionError(f"Drift too high: {drift}")


if __name__ == "__main__":
    run_synaptic_shield_smoke_test()
