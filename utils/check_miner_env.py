#!/usr/bin/env python3
"""
Utility script to check if all required environment variables for the G.O.D miner are set.
This helps diagnose common configuration issues.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from core import constants as cst
from fiber.logging_utils import get_logger

logger = get_logger(__name__)


def check_docker():
    """Check if Docker is available and running."""
    try:
        result = subprocess.run(['docker', 'info'], 
                               capture_output=True, 
                               text=True)
        if result.returncode == 0:
            print("✅ Docker is installed and running")
            return True
        else:
            print("❌ Docker is installed but not running")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed or not in PATH")
        return False


def check_nvidia_docker():
    """Check if NVIDIA Docker runtime is available."""
    try:
        result = subprocess.run(['docker', 'info'], 
                               capture_output=True, 
                               text=True)
        if 'nvidia' in result.stdout:
            print("✅ NVIDIA Docker runtime is available")
            return True
        else:
            print("⚠️ NVIDIA Docker runtime may not be configured")
            return False
    except Exception:
        print("❌ Could not check NVIDIA Docker runtime")
        return False


def check_gpu():
    """Check if NVIDIA GPU is available."""
    try:
        result = subprocess.run(['nvidia-smi'], 
                               capture_output=True, 
                               text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU is available")
            # Extract GPU model
            try:
                for line in result.stdout.splitlines():
                    if '|' in line and 'GeForce' in line or 'Tesla' in line or 'RTX' in line or 'A100' in line:
                        gpu_model = line.split('|')[1].strip()
                        print(f"   GPU Model: {gpu_model}")
                        break
            except Exception:
                pass
            return True
        else:
            print("❌ NVIDIA GPU check failed")
            print(f"Error: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False


def check_env_vars():
    """Check required environment variables."""
    required_vars = {
        'HUGGINGFACE_TOKEN': 'Required for model access and submission',
        'HUGGINGFACE_USERNAME': 'Required for model submission',
        'WANDB_TOKEN': 'Optional for training visualization (not required)',
        'S3_BUCKET_NAME': 'Optional for S3 storage (not required)',
    }
    
    missing = []
    present = []
    
    for var, desc in required_vars.items():
        value = getattr(cst, var, os.environ.get(var))
        if not value:
            missing.append((var, desc))
        else:
            # Mask tokens for privacy
            masked_value = value
            if 'TOKEN' in var and len(value) > 10:
                masked_value = f"{value[:5]}...{value[-5:]}"
            present.append((var, masked_value, desc))
    
    if missing:
        print("\n❌ Missing required environment variables:")
        for var, desc in missing:
            print(f"   {var}: {desc}")
    
    if present:
        print("\n✅ Environment variables present:")
        for var, masked_value, desc in present:
            print(f"   {var}={masked_value}: {desc}")
    
    return len(missing) == 0


def check_directories():
    """Check required directories."""
    required_dirs = {
        'CONFIG_DIR': cst.CONFIG_DIR,
        'OUTPUT_DIR': cst.OUTPUT_DIR,
        'DIFFUSION_DATASET_DIR': cst.DIFFUSION_DATASET_DIR,
    }
    
    for name, path in required_dirs.items():
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            print(f"✅ Directory exists: {name}={abs_path}")
        else:
            print(f"❌ Directory missing: {name}={abs_path}")
            try:
                os.makedirs(abs_path, exist_ok=True)
                print(f"   Created directory: {abs_path}")
            except Exception as e:
                print(f"   Failed to create directory: {e}")


def check_docker_images():
    """Check if required Docker images are available."""
    required_images = {
        'Text Model Miner': cst.MINER_DOCKER_IMAGE,
        'Diffusion Model Miner': cst.MINER_DOCKER_IMAGE_DIFFUSION,
    }
    
    for name, image in required_images.items():
        try:
            result = subprocess.run(['docker', 'image', 'inspect', image], 
                                   capture_output=True, 
                                   text=True)
            if result.returncode == 0:
                print(f"✅ Docker image available: {name} ({image})")
            else:
                print(f"❌ Docker image missing: {name} ({image})")
                print("   You may need to pull the image with:")
                print(f"   docker pull {image}")
        except Exception as e:
            print(f"❌ Failed to check Docker image {image}: {e}")


def main():
    """Run all checks and report results."""
    print("=" * 80)
    print("G.O.D Miner Environment Check")
    print("=" * 80)
    
    all_passed = True
    
    print("\n[1/5] Checking Docker installation:")
    if not check_docker():
        all_passed = False
    
    print("\n[2/5] Checking NVIDIA Docker runtime:")
    if not check_nvidia_docker():
        # Just a warning, not a failure
        pass
    
    print("\n[3/5] Checking NVIDIA GPU:")
    if not check_gpu():
        print("⚠️ WARNING: No NVIDIA GPU detected - miner may not function properly")
        # Not a hard requirement, so don't fail
    
    print("\n[4/5] Checking environment variables:")
    if not check_env_vars():
        all_passed = False
    
    print("\n[5/5] Checking required directories:")
    check_directories()
    
    print("\n[6/5] Checking Docker images:")
    check_docker_images()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ All critical checks passed! The miner should be able to run.")
    else:
        print("❌ Some checks failed. Please fix the issues above before running the miner.")
    print("=" * 80)


if __name__ == "__main__":
    main()