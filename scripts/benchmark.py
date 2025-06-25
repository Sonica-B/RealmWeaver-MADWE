#!/usr/bin/env python3
"""
GPU and Stable Diffusion Benchmark 
"""

import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler


def check_gpu_setup() -> bool:
    """Verify GPU setup for Windows"""
    print("=" * 60)
    print("GPU VERIFICATION")
    print("=" * 60)

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi Processors: {props.multi_processor_count}")

            # Current memory status
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  Currently Allocated: {allocated:.2f} GB")
            print(f"  Currently Reserved: {reserved:.2f} GB")

    return cuda_available


def benchmark_stable_diffusion(output_dir: Path | str = "outputs") -> Dict[str, Any]:
    """Benchmark Stable Diffusion performance"""
    print("\n" + "=" * 60)
    print("STABLE DIFFUSION BENCHMARK")
    print("=" * 60)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # Load pipeline with latest optimizations
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )

    # Use DPM solver for faster inference
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Enable optimizations
    if device == "cuda":
        pipe.enable_attention_slicing()
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled")
        except:
            print("⚠ xformers not available")
        pipe.enable_vae_slicing()

    # Benchmark settings
    prompt = "A fantasy castle on a hill, digital art, highly detailed"
    negative_prompt = "low quality, blurry, pixelated"
    sizes = [(512, 512), (768, 768)]
    steps_list = [20, 30, 50]
    batch_sizes = [1, 2, 4] if device == "cuda" else [1]

    results = {
        "device": device,
        "dtype": str(dtype),
        "timestamp": datetime.now().isoformat(),
        "benchmarks": [],
    }

    # Warmup
    print("\nWarming up...")
    _ = pipe(prompt, num_inference_steps=10, width=512, height=512)

    # Run benchmarks
    for width, height in sizes:
        for steps in steps_list:
            for batch_size in batch_sizes:
                if batch_size > 1 and (width > 512 or height > 512):
                    continue  # Skip large batches for high resolution

                print(
                    f"\nTesting: {width}x{height}, {steps} steps, batch size {batch_size}"
                )

                times = []
                memory_usage = []

                # Create batch prompts
                prompts = [prompt] * batch_size
                negative_prompts = [negative_prompt] * batch_size

                # Run multiple times
                for run in range(3):
                    torch.cuda.empty_cache() if device == "cuda" else None
                    torch.cuda.synchronize() if device == "cuda" else None

                    start = time.perf_counter()

                    images = pipe(
                        prompt=prompts,
                        negative_prompt=negative_prompts,
                        num_inference_steps=steps,
                        guidance_scale=7.5,
                        width=width,
                        height=height,
                    ).images

                    torch.cuda.synchronize() if device == "cuda" else None
                    elapsed = time.perf_counter() - start

                    times.append(elapsed)

                    if device == "cuda":
                        memory_usage.append(torch.cuda.max_memory_allocated() / 1024**3)
                        torch.cuda.reset_peak_memory_stats()

                    print(
                        f"  Run {run+1}: {elapsed:.2f}s ({elapsed/batch_size:.2f}s per image)"
                    )

                # Calculate statistics
                avg_time = np.mean(times)
                std_time = np.std(times)
                avg_time_per_image = avg_time / batch_size
                fps = batch_size / avg_time

                benchmark_result = {
                    "size": f"{width}x{height}",
                    "steps": steps,
                    "batch_size": batch_size,
                    "avg_time": avg_time,
                    "std_time": std_time,
                    "avg_time_per_image": avg_time_per_image,
                    "fps": fps,
                    "memory_gb": np.mean(memory_usage) if memory_usage else None,
                }

                results["benchmarks"].append(benchmark_result)

                print(f"  Average: {avg_time:.2f}s ± {std_time:.2f}s")
                print(f"  Per image: {avg_time_per_image:.2f}s ({fps:.2f} FPS)")
                if memory_usage:
                    print(f"  Peak memory: {np.mean(memory_usage):.2f} GB")

    # Save results
    results_file = (
        output_dir
        / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    best_fps = max(b["fps"] for b in results["benchmarks"])
    best_config = next(b for b in results["benchmarks"] if b["fps"] == best_fps)

    print(f"Best FPS: {best_fps:.2f}")
    print(
        f"Configuration: {best_config['size']}, {best_config['steps']} steps, batch {best_config['batch_size']}"
    )

    return results


def benchmark_memory_usage() -> None:
    """Test memory usage patterns"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory benchmark")
        return

    print("\n" + "=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)

    # Test different model precisions
    precisions = [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16) if torch.cuda.is_bf16_supported() else None,
    ]

    for name, dtype in filter(lambda x: x is not None, precisions):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f"\nTesting {name}:")

        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                use_safetensors=True,
            ).to("cuda")

            # Generate one image
            _ = pipe("test", num_inference_steps=20)

            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            peak = torch.cuda.max_memory_allocated() / 1024**3

            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Peak: {peak:.2f} GB")

            del pipe

        except Exception as e:
            print(f"  Failed: {e}")


def main():
    """Main benchmark function"""
    print("MADWE Project - Performance Benchmark")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")

    # Check GPU
    gpu_available = check_gpu_setup()
    if not gpu_available:
        print("\n⚠ WARNING: CUDA not available! Performance will be limited.")
        response = input("Continue with CPU benchmark? (y/n): ")
        if response.lower() != "y":
            return

    # Run benchmarks
    benchmark_stable_diffusion()

    # Memory analysis (GPU only)
    if gpu_available:
        benchmark_memory_usage()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
