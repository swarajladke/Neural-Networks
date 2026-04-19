import torch
import time
import os
from unittest.mock import patch
from agnis_v4_core import PredictiveHierarchy
from agnis_v4_cognitive import CognitivePredictiveAgent, ThermalGuardian

class MockThermalGuardian(ThermalGuardian):
    """Mocks telemetry to test safety logic without burning a real laptop."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mock_temp = 30
        self.mock_vram = 10.0
        
    def query_telemetry(self):
        return self.mock_temp, self.mock_vram

def test_thermal_safety():
    print("==================================================")
    print(" V5.5: Thermal Safety Guardian Validation")
    print("==================================================")
    
    hierarchy = PredictiveHierarchy([4, 16, 4])
    agent = CognitivePredictiveAgent(hierarchy)
    
    # Replace real guardian with mock
    mock = MockThermalGuardian(device="cpu")
    agent.guardian = mock
    
    x = torch.randn(1, 4)
    y = torch.randn(1, 4)
    
    # 1. Test Normal Operation
    print("\n[Test 1] Normal Operation (30C)")
    mock.mock_temp = 30
    start = time.time()
    agent.observe_and_learn(x, y, max_steps=10)
    print(f"-> Time: {time.time()-start:.4f}s (Expected: fast)")
    
    # 2. Test Caution (70C)
    # The check() runs every 10 calls. We'll force it.
    print("\n[Test 2] Thermal Caution (70C) - Throttling")
    mock.mock_temp = 70
    mock._check_counter = 9 # next call will trigger check
    start = time.time()
    agent.observe_and_learn(x, y, max_steps=10)
    elapsed = time.time() - start
    print(f"-> Time: {elapsed:.4f}s (Expected: ~0.1s delay)")
    if elapsed > 0.1:
        print("[PASS] Throttling detected.")
    else:
        raise AssertionError("No throttling detected at caution temperature.")

    # 3. Test Pause (78C)
    print("\n[Test 3] Thermal Pause (78C) - Mandatory Rest")
    mock.mock_temp = 78
    mock._check_counter = 9
    sleep_calls = []

    def _fake_sleep(seconds):
        sleep_calls.append(float(seconds))

    with patch("agnis_v4_cognitive.time.sleep", side_effect=_fake_sleep):
        agent.observe_and_learn(x, y, max_steps=10)

    print(f"-> Sleep calls captured: {sleep_calls}")
    if 30.0 in sleep_calls and 15.0 in sleep_calls:
        print("[PASS] Mandatory rest path executed without real waiting.")
    else:
        raise AssertionError("Pause path did not trigger expected rest intervals.")

    # 4. Test Emergency (85C)
    print("\n[Test 4] Thermal Emergency (85C) - Auto-Checkpoint")
    mock.mock_temp = 85
    mock._check_counter = 9
    
    checkpoint_path = "thermal_emergency_checkpoint.pt"
    if os.path.exists(checkpoint_path): os.remove(checkpoint_path)
    
    print("-> Triggering emergency (Process should exit)...")
    try:
        # This will call sys.exit(1), which raises SystemExit
        agent.observe_and_learn(x, y, max_steps=10)
    except SystemExit:
        print("-> Caught SystemExit as expected.")
        if os.path.exists(checkpoint_path):
            print(f"[PASS] Emergency checkpoint saved to {checkpoint_path}")
        else:
            raise AssertionError("Checkpoint not saved during emergency shutdown.")
    else:
        raise AssertionError("Thermal emergency did not terminate the process.")

if __name__ == "__main__":
    test_thermal_safety()
