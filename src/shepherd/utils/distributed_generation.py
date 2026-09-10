"""Process-per-GPU generation helpers used by :class:`ShepherdModel`."""
from __future__ import annotations

import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import torch

if TYPE_CHECKING:
    from shepherd.generated_sample import GeneratedSample


def resolve_cuda_devices(devices: int | Sequence[int] | None) -> list[int]:
    """Resolve local CUDA device indices and reject ambiguous selections."""
    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("generate_distributed() requires at least one CUDA GPU")

    if devices is None:
        resolved = list(range(available))
    elif isinstance(devices, int):
        if devices < 1:
            raise ValueError("devices must be a positive count or a sequence of CUDA indices")
        resolved = list(range(devices))
    else:
        resolved = list(devices)

    if not resolved:
        raise ValueError("devices must select at least one CUDA GPU")
    if any(not isinstance(device, int) or isinstance(device, bool) for device in resolved):
        raise TypeError("every CUDA device index must be an integer")
    if len(set(resolved)) != len(resolved):
        raise ValueError("CUDA device indices must be unique")
    if min(resolved) < 0 or max(resolved) >= available:
        raise ValueError(
            f"CUDA device indices must be between 0 and {available - 1}; got {resolved}"
        )
    return resolved


def shard_generate_kwargs(
    generate_kwargs: dict[str, Any],
    num_workers: int,
) -> list[tuple[int, dict[str, Any]]]:
    """Split a total generation batch into contiguous worker-local batches."""
    if "batch_size" not in generate_kwargs:
        raise TypeError("generate_distributed() requires batch_size as a keyword argument")
    total = generate_kwargs["batch_size"]
    if not isinstance(total, int) or isinstance(total, bool) or total < 1:
        raise ValueError("batch_size must be a positive integer")

    worker_count = min(num_workers, total)
    quotient, remainder = divmod(total, worker_count)
    sizes = [quotient + (rank < remainder) for rank in range(worker_count)]

    for name in ("N_x1", "N_x4"):
        value = generate_kwargs.get(name)
        if isinstance(value, list) and len(value) != total:
            raise ValueError(
                f"{name} has {len(value)} entries but total batch_size is {total}"
            )

    shards = []
    start = 0
    requested_verbose = bool(generate_kwargs.get("verbose", True))
    for rank, size in enumerate(sizes):
        local_kwargs = dict(generate_kwargs)
        local_kwargs["batch_size"] = size
        for name in ("N_x1", "N_x4"):
            value = local_kwargs.get(name)
            if isinstance(value, list):
                local_kwargs[name] = value[start:start + size]
        # A single progress bar remains useful; one per worker would interleave output.
        local_kwargs["verbose"] = requested_verbose and rank == 0
        shards.append((start, local_kwargs))
        start += size
    return shards


def distributed_generate_worker(
    rank: int,
    device_ids: list[int],
    checkpoint_path: str,
    shards: list[tuple[int, dict[str, Any]]],
    seed: int,
    generation_method: str,
) -> list[GeneratedSample]:
    """Load one model replica and generate this rank's shard of the total batch."""
    from shepherd.loader import load_model

    device_index = device_ids[rank]
    torch.cuda.set_device(device_index)

    # Offset by the shard's first global sample index. This avoids identical random
    # streams on different ranks while keeping a fixed partition reproducible.
    start, generate_kwargs = shards[rank]
    worker_seed = (seed + start) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed(worker_seed)

    model = load_model(local_checkpoint_path=checkpoint_path, device=f"cuda:{device_index}")
    return getattr(model, generation_method)(**generate_kwargs)


def run_distributed(
    lightning_module,
    generation_method: str,
    generate_kwargs: dict[str, Any],
    *,
    devices: int | Sequence[int] | None,
    seed: int | None,
    checkpoint_path: str | None,
) -> list[GeneratedSample]:
    """Launch process-per-GPU generation workers and concatenate their samples.

    ``checkpoint_path`` must already be resolved by the caller; workers reload the
    model from it rather than inheriting ``lightning_module``, which is moved off
    the GPU for the duration so it does not compete with the replicas.
    """
    if checkpoint_path is None:
        raise ValueError(
            f"{generation_method}_distributed() needs a checkpoint_path because "
            "this ShepherdModel was not created by shepherd.load_model()"
        )
    checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    base_seed = torch.initial_seed() if seed is None else seed
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("seed must be a non-negative integer")

    device_ids = resolve_cuda_devices(devices)
    shards = shard_generate_kwargs(generate_kwargs, len(device_ids))
    try:
        pickle.dumps(shards, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise TypeError(
            f"{generation_method}_distributed() arguments must be serializable "
            "for worker processes"
        ) from error

    # A small batch can yield fewer shards than devices; the extra GPUs go unused.
    device_ids = device_ids[:len(shards)]

    original_device = lightning_module.device
    moved_from_cuda = original_device.type == "cuda"
    if moved_from_cuda:
        lightning_module.to("cpu")
        lightning_module.model.device = torch.device("cpu")
        torch.cuda.empty_cache()

    try:
        # Spawn is required for CUDA; the executor drains each result on its own
        # thread, so large shards cannot deadlock the way a queue plus join would.
        with ProcessPoolExecutor(
            max_workers=len(shards),
            mp_context=torch.multiprocessing.get_context("spawn"),
        ) as pool:
            futures = [
                pool.submit(
                    distributed_generate_worker,
                    rank,
                    device_ids,
                    checkpoint_path,
                    shards,
                    base_seed,
                    generation_method,
                )
                for rank in range(len(shards))
            ]
            # Ranks are consumed in order so samples stay in global batch order.
            return [sample for future in futures for sample in future.result()]
    finally:
        if moved_from_cuda:
            lightning_module.to(original_device)
            lightning_module.model.device = original_device
