# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Demonstrates reinforcement learning from human feedback (RLHF) using vLLM and Ray.

The script separates training and inference workloads onto distinct GPUs
so that Ray can manage process placement and inter-process communication.
A Hugging Face Transformer model occupies GPU 0 for training, whereas a
tensor-parallel vLLM inference engine occupies GPU 1–2.

The example performs the following steps:

* Load the training model on GPU 0.
* Split the inference model across GPUs 1–2 using vLLM's tensor parallelism
  and Ray placement groups.
* Generate text from a list of prompts using the inference engine.
* Update the weights of the training model and broadcast the updated weights
  to the inference engine by using a Ray collective RPC group. Note that
  for demonstration purposes we simply zero out the weights.

For a production-ready implementation that supports multiple training and
inference replicas, see the OpenRLHF framework:
https://github.com/OpenRLHF/OpenRLHF

This example assumes a single-node cluster with three GPUs, but Ray
supports multi-node clusters. vLLM expects the GPUs are only used for vLLM
workloads. Residual GPU activity interferes with vLLM memory profiling and
causes unexpected behavior.
"""

import asyncio
import ipaddress
import uuid
from typing import Any

import ray
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    Store,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)
from transformers import AutoModelForCausalLM

import vllm
from vllm import SamplingParams
from vllm.utils.network_utils import get_ip, get_open_port


def get_tcp_url(host: str, port: int) -> str:
    """
    Formats the TCP URL for the given host and port,
    handling IPv6 addresses correctly.
    Args:
        host (str): The hostname or IP address.
        port (int): The port number.
    Returns:
        str: The formatted TCP URL.
    """
    try:
        if isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address):
            return f"tcp://[{host}]:{port}"
    except ValueError:
        # not a literal IP, probably a hostname
        pass
    return f"tcp://{host}:{port}"


def init_custom_process_group(
    backend: str | Backend = None,
    init_method: str | None = None,
    timeout: Any | None = None,
    world_size: int = -1,
    rank: int = -1,
    store: Store | None = None,
    group_name: str = None,
    pg_options: Any | None = None,
):
    print(f"DEBUG: init_custom_process_group rank={rank}/{world_size}")
    assert (store is None) or (init_method is None), (
        "Cannot specify both init_method and store."
    )

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    By defining an extension class, the code can work no matter what is
    the underlying worker class.

    NOTE: we define this class in a separate module, and the main module
    should pass the full qualified name as `worker_extension_cls` argument.
    """

    def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size
    ):
        from vllm.distributed.parallel_state import get_world_group

        rank = get_world_group().rank + rank_offset
        self.model_update_group = init_custom_process_group(
            backend="nccl",
            init_method=get_tcp_url(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name="vllm_weight_update_group",
        )

    def update_weight(self, name, dtype_name, shape):
        dtype = getattr(torch, dtype_name)
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, src=0, group=self.model_update_group)

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed(self):
        """
        Check if the weights are updated to 0.
        """
        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated


class MyLLM:
    """Configure the vLLM worker for Ray placement group execution."""

    def __init__(self, *args, **kwargs):
        # Remove the top-level CUDA_VISIBLE_DEVICES variable set by Ray
        # so that vLLM can manage its own device placement within the worker.
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        self.engine = vllm.AsyncLLMEngine.from_engine_args(
            vllm.AsyncEngineArgs(
                model="facebook/opt-125m",
                enforce_eager=True,
                worker_extension_cls="rl_fully_async.WorkerExtension",
                tensor_parallel_size=2,
                distributed_executor_backend="ray",
            )
        )
        self.generation_paused_event = asyncio.Event()

    async def generate_batch(
        self, prompts: list[str], sampling_params: vllm.SamplingParams
    ) -> list[vllm.RequestOutput]:
        return await asyncio.gather(
            *[self.generate(prompt, sampling_params) for prompt in prompts]
        )

    async def generate(
        self, prompt: str, sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        async for request_output in self.engine.generate(
            prompt, sampling_params, request_id=str(uuid.uuid4())
        ):
            final_output = request_output
        # print(f"DEBUG: generated {final_output.outputs}, length {len(final_output.outputs)}")
        return final_output

    async def generate_with_retry(
        self, prompt: str, sampling_params: vllm.SamplingParams
    ) -> vllm.RequestOutput:
        finish_reason = "abort"
        while finish_reason == "abort":
            await self._wait_for_generation_to_resume()
            output = await self.generate(prompt, sampling_params)
            finish_reason = output.outputs[0].finish_reason
            if finish_reason == "abort":
                print(f"REQ ABORTED, prompt: {prompt}, text: {output.outputs[0].text}")
            prompt += output.outputs[0].text
        return output

    async def abort_generation(self) -> None:
        unfinished_request_ids = list(
            self.engine.output_processor.request_states.keys()
        )
        if unfinished_request_ids:
            await self.engine.abort(unfinished_request_ids)
        await self.engine.reset_prefix_cache()
        print(
            f"abort_generation() finished, aborted {len(unfinished_request_ids)} requests"
        )

    async def resume_generation(self) -> None:
        self.generation_paused_event.clear()

    async def collective_rpc(self, method: str, args: tuple = ()):
        return await self.engine.collective_rpc(method, args=args)

    async def _wait_for_generation_to_resume(self) -> None:
        """Waits for generation to be resumed, intended for in-flight weight updates and partial rollouts."""
        while self.generation_paused_event.is_set():
            await asyncio.sleep(0.5)


@ray.remote(num_gpus=1)
def main():
    # Load the OPT-125M model onto GPU 0 for the training workload.
    train_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
    train_model.to("cuda:0")

    # Create a placement group that reserves GPU 1–2 for the vLLM inference engine.
    # Learn more about Ray placement groups:
    # https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html
    pg_inference = placement_group([{"GPU": 1, "CPU": 0}] * 2)
    ray.get(pg_inference.ready())
    scheduling_inference = PlacementGroupSchedulingStrategy(
        placement_group=pg_inference,
        placement_group_capture_child_tasks=True,
        placement_group_bundle_index=0,
    )

    # Launch the vLLM inference engine. The `enforce_eager` flag reduces
    # start-up latency.
    llm = ray.remote(
        num_cpus=0,
        num_gpus=0,
        scheduling_strategy=scheduling_inference,
    )(MyLLM).remote(
        model="facebook/opt-125m",
        enforce_eager=True,
        worker_extension_cls="rlhf_utils.WorkerExtension",
        tensor_parallel_size=2,
        distributed_executor_backend="ray",
    )

    # Set up the communication channel between the training process and the
    # inference engine.
    master_address = get_ip()
    master_port = get_open_port()

    handle = llm.collective_rpc.remote(
        "init_weight_update_group", args=(master_address, master_port, 1, 3)
    )

    model_update_group = init_custom_process_group(
        backend="nccl",
        init_method=get_tcp_url(master_address, master_port),
        world_size=3,
        rank=0,
        group_name="vllm_weight_update_group",
    )
    ray.get(handle)

    # Generate text from the prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0)

    # Submit all generation requests asynchronously using ray remote
    pending_futures = [
        llm.generate_with_retry.remote(prompt, sampling_params) for prompt in prompts
    ]
    all_outputs = []

    # Wait for one result to finish (variable time per request)
    finished, pending_futures = ray.wait(pending_futures, num_returns=1)

    # Get the finished result
    output = ray.get(finished[0])
    all_outputs.append(output)

    # Pause generation by calling abort_generation
    ray.get(llm.abort_generation.remote())
    print(f"Prompt: {output.prompt!r}\nGenerated text: {output.outputs[0].text!r}")
    print(f"Stop reason: {output.outputs[0].finish_reason!r}")

    # Simulate a training step by zeroing out all model weights.
    # In a real RLHF training loop the weights would be updated using the gradient
    # from an RL objective such as PPO on a reward model.
    for name, p in train_model.named_parameters():
        p.data.zero_()

    # Synchronize the updated weights to the inference engine.
    for name, p in train_model.named_parameters():
        dtype_name = str(p.dtype).split(".")[-1]
        handle = llm.collective_rpc.remote(
            "update_weight", args=(name, dtype_name, p.shape)
        )
        torch.distributed.broadcast(p.data, src=0, group=model_update_group)
        ray.get(handle)

    # Verify that the inference weights have been updated.
    assert all(ray.get(llm.collective_rpc.remote("check_weights_changed")))

    # Resume generation
    ray.get(llm.resume_generation.remote())

    # Wait for all remaining results
    remaining_results = ray.get(list(pending_futures))
    all_outputs.extend(remaining_results)

    # Print out the results
    print("-" * 50)
    for output in all_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print(f"Stop reason: {output.outputs[0]!r}")
        print("-" * 50)


if __name__ == "__main__":
    ray.init()
    ray.get(main.remote())
