"""
Model optimization pipeline for MADWE
Implements ONNX export, TensorRT optimization, and performance benchmarking
Day 4: Thursday, June 6 - Integration Framework
"""

import torch
import onnx
import onnxruntime as ort
import tensorrt as trt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import time
import json
from dataclasses import dataclass
from tqdm import tqdm
import pycuda.driver as cuda
import pycuda.autoinit


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""

    fp16_mode: bool = True
    int8_mode: bool = False
    max_workspace_size: int = 2 << 30  # 2GB
    max_batch_size: int = 1
    optimize_for_rtx3060: bool = True
    enable_cudnn: bool = True
    profile_layers: bool = True

    # RTX 3060 specific optimizations
    use_tensor_cores: bool = True
    enable_memory_pooling: bool = True
    use_cuda_graphs: bool = True


class ModelOptimizer:
    """Handles model optimization for real-time inference"""

    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.benchmark_results = {}

        # TensorRT logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)

        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        if self.cuda_available:
            self.device_name = torch.cuda.get_device_name(0)
            self.device_capability = torch.cuda.get_device_capability(0)
            print(f"CUDA Device: {self.device_name}")
            print(f"Compute Capability: {self.device_capability}")

    def export_to_onnx(
        self,
        model: torch.nn.Module,
        dummy_input: torch.Tensor,
        output_path: Path,
        input_names: List[str],
        output_names: List[str],
        dynamic_axes: Dict[str, Dict[int, str]] = None,
    ) -> Path:
        """Export PyTorch model to ONNX format"""
        print(f"Exporting model to ONNX: {output_path}")

        # Ensure model is in eval mode
        model.eval()

        # Export with optimizations
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=16,  # Latest ONNX opset
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            verbose=False,
        )

        # Verify ONNX model
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)

        # Optimize ONNX model
        from onnx import optimizer

        optimized_model = optimizer.optimize(onnx_model)
        onnx.save(optimized_model, str(output_path))

        print(f"ONNX export successful: {output_path}")
        return output_path

    def export_unet_to_onnx(self, unet: UNet2DConditionModel, output_dir: Path) -> Path:
        """Export UNet model to ONNX with proper inputs"""
        output_path = output_dir / "unet_optimized.onnx"

        # Create dummy inputs
        batch_size = 1
        latent_channels = 4
        latent_size = 64  # 512/8 for VAE encoding

        dummy_latents = torch.randn(
            batch_size, latent_channels, latent_size, latent_size
        )
        dummy_timestep = torch.tensor([1])
        dummy_encoder_hidden_states = torch.randn(
            batch_size, 77, 768
        )  # CLIP encoding size

        # Prepare model
        unet.eval()
        if self.config.fp16_mode and self.cuda_available:
            unet = unet.half().cuda()
            dummy_latents = dummy_latents.half().cuda()
            dummy_encoder_hidden_states = dummy_encoder_hidden_states.half().cuda()

        # Define dynamic axes for variable batch size
        dynamic_axes = {
            "latent_model_input": {0: "batch_size"},
            "timestep": {0: "batch_size"},
            "encoder_hidden_states": {0: "batch_size"},
            "noise_pred": {0: "batch_size"},
        }

        # Export
        return self.export_to_onnx(
            model=unet,
            dummy_input=(dummy_latents, dummy_timestep, dummy_encoder_hidden_states),
            output_path=output_path,
            input_names=["latent_model_input", "timestep", "encoder_hidden_states"],
            output_names=["noise_pred"],
            dynamic_axes=dynamic_axes,
        )

    def optimize_with_tensorrt(
        self, onnx_path: Path, output_path: Path
    ) -> Optional[Path]:
        """Optimize ONNX model with TensorRT"""
        if not self.cuda_available:
            print("CUDA not available, skipping TensorRT optimization")
            return None

        print(f"Optimizing with TensorRT: {onnx_path}")

        # Create builder
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.trt_logger)

        # Parse ONNX model
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size

        # RTX 3060 optimizations
        if self.config.optimize_for_rtx3060:
            # Enable FP16 mode (RTX 3060 has good FP16 performance)
            if self.config.fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)

            # Enable INT8 mode if calibration data is available
            if self.config.int8_mode:
                config.set_flag(trt.BuilderFlag.INT8)

            # Enable tensor cores
            if self.config.use_tensor_cores:
                config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

            # Memory pooling for better memory management
            if self.config.enable_memory_pooling:
                config.set_memory_pool_limit(
                    trt.MemoryPoolType.WORKSPACE, self.config.max_workspace_size
                )

        # Create optimization profile
        profile = builder.create_optimization_profile()

        # Set dynamic shapes for inputs
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            shape = input_tensor.shape

            # Set min, opt, max shapes for dynamic dimensions
            min_shape = [1 if dim == -1 else dim for dim in shape]
            opt_shape = [
                self.config.max_batch_size if dim == -1 else dim for dim in shape
            ]
            max_shape = [
                4 if dim == -1 else dim for dim in shape
            ]  # Allow up to 4 batch size

            profile.set_shape(
                input_tensor.name, min=min_shape, opt=opt_shape, max=max_shape
            )

        config.add_optimization_profile(profile)

        # Build engine
        print("Building TensorRT engine (this may take several minutes)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            print("Failed to build TensorRT engine")
            return None

        # Serialize engine
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        print(f"TensorRT engine saved: {output_path}")
        return output_path

    def benchmark_inference(
        self,
        model_path: Path,
        model_type: str = "onnx",
        num_runs: int = 100,
        warmup_runs: int = 10,
    ) -> Dict[str, Any]:
        """Benchmark model inference performance"""
        print(f"\nBenchmarking {model_type} model: {model_path}")

        results = {
            "model_type": model_type,
            "model_path": str(model_path),
            "num_runs": num_runs,
            "warmup_runs": warmup_runs,
            "times": [],
            "memory_usage": [],
        }

        if model_type == "onnx":
            results.update(self._benchmark_onnx(model_path, num_runs, warmup_runs))
        elif model_type == "tensorrt":
            results.update(self._benchmark_tensorrt(model_path, num_runs, warmup_runs))
        elif model_type == "pytorch":
            results.update(self._benchmark_pytorch(model_path, num_runs, warmup_runs))

        # Calculate statistics
        times = np.array(results["times"])
        results["stats"] = {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "p50_time": np.percentile(times, 50),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99),
            "fps": 1.0 / np.mean(times) if np.mean(times) > 0 else 0,
        }

        return results

    def _benchmark_onnx(
        self, model_path: Path, num_runs: int, warmup_runs: int
    ) -> Dict[str, Any]:
        """Benchmark ONNX model"""
        # Create ONNX Runtime session
        providers = (
            ["CUDAExecutionProvider"]
            if self.cuda_available
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(model_path), providers=providers)

        # Get input details
        input_details = session.get_inputs()

        # Create dummy inputs
        inputs = {}
        for input_detail in input_details:
            shape = input_detail.shape
            # Replace dynamic dimensions with concrete values
            shape = [
                self.config.max_batch_size if dim == "batch_size" or dim == -1 else dim
                for dim in shape
            ]

            dtype = np.float16 if self.config.fp16_mode else np.float32
            inputs[input_detail.name] = np.random.randn(*shape).astype(dtype)

        times = []
        memory_usage = []

        # Warmup
        print(f"Warming up with {warmup_runs} runs...")
        for _ in range(warmup_runs):
            _ = session.run(None, inputs)

        # Benchmark
        print(f"Benchmarking with {num_runs} runs...")
        for _ in tqdm(range(num_runs)):
            if self.cuda_available:
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()

            start_time = time.perf_counter()
            _ = session.run(None, inputs)

            if self.cuda_available:
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            times.append(end_time - start_time)

            if self.cuda_available:
                end_mem = torch.cuda.memory_allocated()
                memory_usage.append(end_mem - start_mem)

        return {
            "times": times,
            "memory_usage": memory_usage,
            "input_shapes": {name: list(inp.shape) for name, inp in inputs.items()},
        }

    def _benchmark_tensorrt(
        self, engine_path: Path, num_runs: int, warmup_runs: int
    ) -> Dict[str, Any]:
        """Benchmark TensorRT engine"""
        if not self.cuda_available:
            return {"error": "CUDA not available for TensorRT"}

        # Load engine
        runtime = trt.Runtime(self.trt_logger)
        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()

        # Allocate buffers
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        # Fill input data
        for inp in inputs:
            np.copyto(
                inp["host"],
                np.random.randn(*inp["host"].shape).astype(inp["host"].dtype),
            )

        times = []

        # Warmup
        for _ in range(warmup_runs):
            # Transfer input data to GPU
            for inp in inputs:
                cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            # Transfer predictions back
            for out in outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

            stream.synchronize()

        # Benchmark
        for _ in tqdm(range(num_runs)):
            start_time = time.perf_counter()

            # Transfer input data to GPU
            for inp in inputs:
                cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

            # Transfer predictions back
            for out in outputs:
                cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

            stream.synchronize()

            end_time = time.perf_counter()
            times.append(end_time - start_time)

        return {"times": times}

    def compare_optimization_methods(
        self, base_model_path: Path, output_dir: Path
    ) -> Dict[str, Any]:
        """Compare different optimization methods"""
        results = {}

        # Load base model (assume it's a UNet checkpoint)
        print("Loading base model...")
        unet = UNet2DConditionModel.from_pretrained(base_model_path)

        # 1. Benchmark PyTorch baseline
        print("\n1. Benchmarking PyTorch baseline...")
        pytorch_results = self._benchmark_pytorch_unet(unet, num_runs=50)
        results["pytorch"] = pytorch_results

        # 2. Export and benchmark ONNX
        print("\n2. Exporting to ONNX and benchmarking...")
        onnx_path = self.export_unet_to_onnx(unet, output_dir)
        onnx_results = self.benchmark_inference(
            onnx_path, model_type="onnx", num_runs=50
        )
        results["onnx"] = onnx_results

        # 3. Optimize with TensorRT and benchmark
        if self.cuda_available:
            print("\n3. Optimizing with TensorRT and benchmarking...")
            trt_path = output_dir / "unet_tensorrt.engine"
            trt_result = self.optimize_with_tensorrt(onnx_path, trt_path)

            if trt_result:
                trt_results = self.benchmark_inference(
                    trt_path, model_type="tensorrt", num_runs=50
                )
                results["tensorrt"] = trt_results

        # 4. Compare results
        comparison = self._generate_comparison_report(results)

        # Save results
        results_path = output_dir / "optimization_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Save comparison report
        report_path = output_dir / "optimization_report.txt"
        with open(report_path, "w") as f:
            f.write(comparison)

        print(f"\nResults saved to: {results_path}")
        print(f"Report saved to: {report_path}")

        return results

    def _benchmark_pytorch_unet(
        self, unet: UNet2DConditionModel, num_runs: int = 50
    ) -> Dict[str, Any]:
        """Benchmark PyTorch UNet"""
        unet.eval()

        if self.cuda_available:
            unet = unet.cuda()
            if self.config.fp16_mode:
                unet = unet.half()

        # Create dummy inputs
        batch_size = self.config.max_batch_size
        dummy_latents = torch.randn(batch_size, 4, 64, 64)
        dummy_timestep = torch.tensor([1])
        dummy_encoder_hidden_states = torch.randn(batch_size, 77, 768)

        if self.cuda_available:
            dummy_latents = dummy_latents.cuda()
            dummy_timestep = dummy_timestep.cuda()
            dummy_encoder_hidden_states = dummy_encoder_hidden_states.cuda()

            if self.config.fp16_mode:
                dummy_latents = dummy_latents.half()
                dummy_encoder_hidden_states = dummy_encoder_hidden_states.half()

        times = []
        memory_usage = []

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = unet(dummy_latents, dummy_timestep, dummy_encoder_hidden_states)

        # Benchmark
        for _ in tqdm(range(num_runs)):
            if self.cuda_available:
                torch.cuda.synchronize()
                start_mem = torch.cuda.memory_allocated()

            start_time = time.perf_counter()

            with torch.no_grad():
                _ = unet(dummy_latents, dummy_timestep, dummy_encoder_hidden_states)

            if self.cuda_available:
                torch.cuda.synchronize()

            end_time = time.perf_counter()

            times.append(end_time - start_time)

            if self.cuda_available:
                end_mem = torch.cuda.memory_allocated()
                memory_usage.append(
                    (end_mem - start_mem) / 1024 / 1024
                )  # Convert to MB

        return {
            "times": times,
            "memory_usage": memory_usage,
            "stats": {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "fps": 1.0 / np.mean(times),
            },
        }

    def _generate_comparison_report(self, results: Dict[str, Any]) -> str:
        """Generate comparison report"""
        report = "MADWE Model Optimization Report\n"
        report += "=" * 50 + "\n\n"

        report += f"Device: {self.device_name if self.cuda_available else 'CPU'}\n"
        report += f"FP16 Mode: {self.config.fp16_mode}\n"
        report += f"Batch Size: {self.config.max_batch_size}\n\n"

        # Performance comparison
        report += "Performance Comparison:\n"
        report += "-" * 30 + "\n"

        for method, result in results.items():
            if "stats" in result:
                stats = result["stats"]
                report += f"\n{method.upper()}:\n"
                report += f"  Mean inference time: {stats['mean_time']*1000:.2f} ms\n"
                report += f"  Std deviation: {stats['std_time']*1000:.2f} ms\n"
                report += f"  FPS: {stats['fps']:.1f}\n"

                if "memory_usage" in result and result["memory_usage"]:
                    avg_mem = np.mean(result["memory_usage"])
                    report += f"  Avg memory usage: {avg_mem:.1f} MB\n"

        # Speedup analysis
        if "pytorch" in results and "stats" in results["pytorch"]:
            pytorch_time = results["pytorch"]["stats"]["mean_time"]
            report += "\nSpeedup vs PyTorch:\n"
            report += "-" * 30 + "\n"

            for method, result in results.items():
                if method != "pytorch" and "stats" in result:
                    speedup = pytorch_time / result["stats"]["mean_time"]
                    report += f"{method}: {speedup:.2f}x\n"

        return report


def main():
    """Main optimization pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Optimize models for MADWE")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/optimized",
        help="Output directory",
    )
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--int8", action="store_true", help="Use INT8 quantization")
    parser.add_argument(
        "--benchmark-only", action="store_true", help="Only run benchmarks"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure optimization
    config = OptimizationConfig(fp16_mode=args.fp16, int8_mode=args.int8)

    optimizer = ModelOptimizer(config)

    if args.benchmark_only:
        # Just benchmark existing model
        results = optimizer.benchmark_inference(
            Path(args.model_path),
            model_type="onnx" if args.model_path.endswith(".onnx") else "pytorch",
        )
        print(f"\nBenchmark Results:")
        print(f"Mean time: {results['stats']['mean_time']*1000:.2f} ms")
        print(f"FPS: {results['stats']['fps']:.1f}")
    else:
        # Run full optimization pipeline
        results = optimizer.compare_optimization_methods(
            Path(args.model_path), output_dir
        )

        print("\nOptimization complete!")
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
