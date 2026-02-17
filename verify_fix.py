
import os
import sys
from unittest.mock import patch

# Ensure we can import the local package
sys.path.insert(0, os.path.abspath("src"))

from mqt.yaqs.simulator import available_cpus

def test_default_cap():
    # Unset env var if present
    if "YAQS_MAX_WORKERS" in os.environ:
        del os.environ["YAQS_MAX_WORKERS"]
    
    # Mock cpu_count to return a large number (e.g. 128)
    with patch("os.cpu_count", return_value=128):
        # We also need to patch multiprocessing.cpu_count just in case, though the code uses os.cpu_count first
        with patch("multiprocessing.cpu_count", return_value=128):
             cpus = available_cpus()
             print(f"Default with 128 cores: {cpus}")
             assert cpus == 61, f"Expected 61, got {cpus}"

def test_env_override():
    # Set limit to 10
    os.environ["YAQS_MAX_WORKERS"] = "10"
    
    with patch("os.cpu_count", return_value=128):
        cpus = available_cpus()
        print(f"Override=10 with 128 cores: {cpus}")
        assert cpus == 10, f"Expected 10, got {cpus}"
        
    # Set limit to 100
    os.environ["YAQS_MAX_WORKERS"] = "100"
    with patch("os.cpu_count", return_value=128):
        cpus = available_cpus()
        print(f"Override=100 with 128 cores: {cpus}")
        assert cpus == 100, f"Expected 100, got {cpus}"
        
    # Clean up
    del os.environ["YAQS_MAX_WORKERS"]

def test_real_hardware_respect():
    # Ensure we don't exceed physical cores even if limit is high
    real_cores = os.cpu_count() or 1
    os.environ["YAQS_MAX_WORKERS"] = str(real_cores + 50)
    cpus = available_cpus()
    print(f"Override={real_cores+50} with real cores ({real_cores}): {cpus}")
    assert cpus <= real_cores, f"Expected <= {real_cores}, got {cpus}"
    
    # Clean up
    del os.environ["YAQS_MAX_WORKERS"]

if __name__ == "__main__":
    try:
        test_default_cap()
        test_env_override()
        test_real_hardware_respect()
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
