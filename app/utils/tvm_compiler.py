"""
TVM compilation utilities for model optimization.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
from loguru import logger


def compile_with_tvm(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    target: str = "llvm",
    opt_level: int = 3,
    output_path: Optional[str] = None
) -> Optional[object]:
    """
    Compile PyTorch model using TVM for optimized inference.

    Args:
        model: PyTorch model to compile
        input_shape: Shape of input tensor
        target: TVM target (e.g., "llvm", "llvm -mcpu=core-avx2", "cuda")
        opt_level: Optimization level (0-3)
        output_path: Optional path to save compiled model

    Returns:
        TVM runtime module or None if compilation failed
    """
    try:
        import tvm
        from tvm import relay
        import tvm.relay.testing

        logger.info(f"Compiling model with TVM for target: {target}")

        # Set model to evaluation mode
        model.eval()

        # Create example input
        input_data = torch.randn(*input_shape)

        # Trace the model
        traced_model = torch.jit.trace(model, input_data)

        # Convert to TVM Relay
        input_name = "input"
        shape_list = [(input_name, input_shape)]

        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

        # Build with TVM
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)

        # Save compiled model if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            lib.export_library(output_path)
            logger.info(f"TVM compiled model saved: {output_path}")

        logger.info("TVM compilation successful")
        return lib

    except ImportError:
        logger.error("TVM not installed. Install with: pip install apache-tvm")
        return None
    except Exception as e:
        logger.error(f"TVM compilation failed: {e}")
        return None


def auto_tune_tvm_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    target: str = "llvm",
    n_trials: int = 1000,
    output_path: Optional[str] = None
) -> Optional[object]:
    """
    Auto-tune TVM model for optimal performance.

    Args:
        model: PyTorch model to tune
        input_shape: Shape of input tensor
        target: TVM target
        n_trials: Number of tuning trials
        output_path: Optional path to save tuned model

    Returns:
        Tuned TVM runtime module or None
    """
    try:
        import tvm
        from tvm import relay, autotvm
        from tvm.autotvm.tuner import XGBTuner

        logger.info(f"Auto-tuning TVM model with {n_trials} trials...")

        # Set model to evaluation mode
        model.eval()

        # Create example input
        input_data = torch.randn(*input_shape)

        # Trace the model
        traced_model = torch.jit.trace(model, input_data)

        # Convert to TVM Relay
        input_name = "input"
        shape_list = [(input_name, input_shape)]

        mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

        # Extract tuning tasks
        tasks = autotvm.task.extract_from_program(
            mod["main"],
            target=target,
            params=params
        )

        # Tune each task
        for i, task in enumerate(tasks):
            logger.info(f"Tuning task {i+1}/{len(tasks)}: {task.name}")

            # Create tuner
            tuner = XGBTuner(task)

            # Tune
            tuner.tune(
                n_trial=min(n_trials, len(task.config_space)),
                measure_option=autotvm.measure_option(
                    builder=autotvm.LocalBuilder(),
                    runner=autotvm.LocalRunner(
                        number=10,
                        repeat=1,
                        timeout=4,
                        min_repeat_ms=150
                    )
                )
            )

        # Build with best configs
        with autotvm.apply_history_best("tuning_log.json"):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)

        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            lib.export_library(output_path)
            logger.info(f"Tuned model saved: {output_path}")

        logger.info("Auto-tuning completed successfully")
        return lib

    except ImportError:
        logger.error("TVM not installed or missing dependencies")
        return None
    except Exception as e:
        logger.error(f"Auto-tuning failed: {e}")
        return None


def benchmark_tvm_model(
    lib: object,
    input_shape: Tuple[int, ...],
    target: str = "llvm",
    n_iterations: int = 100
) -> dict:
    """
    Benchmark TVM compiled model performance.

    Args:
        lib: TVM runtime module
        input_shape: Shape of input tensor
        target: TVM target
        n_iterations: Number of benchmark iterations

    Returns:
        Dictionary with benchmark metrics
    """
    try:
        import tvm
        from tvm.contrib import graph_executor
        import time

        logger.info(f"Benchmarking TVM model ({n_iterations} iterations)...")

        # Create runtime
        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))

        # Create random input
        input_data = np.random.randn(*input_shape).astype("float32")
        module.set_input("input", input_data)

        # Warm up
        for _ in range(10):
            module.run()

        # Benchmark
        start_time = time.time()
        for _ in range(n_iterations):
            module.run()
        end_time = time.time()

        avg_time = (end_time - start_time) / n_iterations

        metrics = {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': 1.0 / avg_time,
            'iterations': n_iterations,
            'target': target
        }

        logger.info(f"Average inference time: {metrics['avg_inference_time_ms']:.3f} ms")
        logger.info(f"Throughput: {metrics['throughput_samples_per_sec']:.2f} samples/sec")

        return metrics

    except Exception as e:
        logger.error(f"Benchmarking failed: {e}")
        return {}


def compile_for_multiple_targets(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    targets: List[str] = ["llvm", "llvm -mcpu=core-avx2"],
    output_dir: str = "./models/compiled"
) -> dict:
    """
    Compile model for multiple hardware targets.

    Args:
        model: PyTorch model to compile
        input_shape: Shape of input tensor
        targets: List of TVM targets
        output_dir: Directory to save compiled models

    Returns:
        Dictionary mapping targets to compiled modules
    """
    compiled_models = {}

    for target in targets:
        logger.info(f"Compiling for target: {target}")

        # Create safe filename
        target_name = target.replace(" ", "_").replace("-", "_")
        output_path = str(Path(output_dir) / f"model_{target_name}.so")

        # Compile
        lib = compile_with_tvm(
            model,
            input_shape,
            target=target,
            output_path=output_path
        )

        if lib is not None:
            compiled_models[target] = {
                'module': lib,
                'path': output_path
            }

            # Benchmark
            metrics = benchmark_tvm_model(lib, input_shape, target)
            compiled_models[target]['metrics'] = metrics

    return compiled_models


class TVMCompiler:
    """
    Helper class for TVM compilation workflow.
    """

    def __init__(self, model: torch.nn.Module, input_shape: Tuple[int, ...]):
        """
        Initialize TVM compiler.

        Args:
            model: PyTorch model to compile
            input_shape: Shape of input tensor
        """
        self.model = model
        self.input_shape = input_shape
        self.compiled_lib = None
        self.target = None

    def compile(
        self,
        target: str = "llvm",
        opt_level: int = 3,
        output_path: Optional[str] = None
    ) -> bool:
        """
        Compile model with TVM.

        Args:
            target: TVM target
            opt_level: Optimization level
            output_path: Optional output path

        Returns:
            True if successful
        """
        self.compiled_lib = compile_with_tvm(
            self.model,
            self.input_shape,
            target=target,
            opt_level=opt_level,
            output_path=output_path
        )
        self.target = target
        return self.compiled_lib is not None

    def benchmark(self, n_iterations: int = 100) -> dict:
        """
        Benchmark compiled model.

        Args:
            n_iterations: Number of iterations

        Returns:
            Benchmark metrics
        """
        if self.compiled_lib is None:
            logger.error("Model not compiled yet")
            return {}

        return benchmark_tvm_model(
            self.compiled_lib,
            self.input_shape,
            self.target,
            n_iterations
        )

    def optimize(self, n_trials: int = 1000) -> bool:
        """
        Auto-tune model for optimal performance.

        Args:
            n_trials: Number of tuning trials

        Returns:
            True if successful
        """
        self.compiled_lib = auto_tune_tvm_model(
            self.model,
            self.input_shape,
            target=self.target or "llvm",
            n_trials=n_trials
        )
        return self.compiled_lib is not None
