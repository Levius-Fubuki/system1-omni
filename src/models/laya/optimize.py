"""What the Laya worker changes about the model itself to make it faster on the GPU.

- fp16 weights: keep the checkpoint's own precision instead of laya's fp32 upcast.
- Compile: torch.compile for the batches where it pays off on MPS.

Both go through one wrapper module that replaces `agent.model`, so that on the CPU the worker always runs
laya's own fp32 model. worker.py decides when to apply them and reports the result in /health.
"""

from typing import Any


def _served(agent: Any) -> Any:
    """The module the worker puts in place of laya's model, created on first use.

    On the GPU it runs the compiled paths when there are any, otherwise laya's model. On the CPU it always
    runs laya's model in fp32: laya moves the model to the CPU when a request runs out of GPU memory, and
    there fp16 weights are slower (334 ms against 138 ms for a 68-token request) and the compiled graphs
    would first recompile (28 s measured). So after a fallback the worker behaves like plain laya.
    """
    import torch

    if getattr(agent.model, "laya_worker_wrapper", False):
        return agent.model
    eager = agent.model

    class Served(torch.nn.Module):
        laya_worker_wrapper = True

        def __init__(self):
            super().__init__()
            self.eager = eager
            self.fp16 = False
            self.paths = None  # (whole model compiled, encoder-only compiled); a tuple is not a submodule

        def forward(self, input_ids, *args, **kwargs):
            if input_ids.device.type == "cpu":
                if self.fp16:
                    self.eager.float()
                    self.fp16 = False
                return self.eager(input_ids, *args, **kwargs)
            if self.paths is None:
                return self.eager(input_ids, *args, **kwargs)
            whole, encoder_only = self.paths
            return (whole if input_ids.shape[0] == 1 else encoder_only)(input_ids, *args, **kwargs)

    agent.model = Served()
    return agent.model


def use_fp16_weights(agent: Any) -> None:
    """Keep the weights in fp16, the checkpoint's own precision, so the conversion is exact. laya 0.3.20
    upcasts them to fp32 on MPS and CPU. `act_head` stays fp32 because laya feeds it `.float()` features.
    For the GPU only: see _served for what happens on the CPU."""
    served = _served(agent)
    served.eager.half()
    act_head = getattr(served.eager, "act_head", None)
    if act_head is not None:
        act_head.float()
    served.fp16 = True


def compile_agent(agent: Any) -> None:
    """Compile the model for the batches where it pays off on MPS, sharing the same parameters.

    A batch of one row (one question) runs the whole model compiled. A batch of several rows runs only
    the encoder compiled and laya's decision head eagerly: the head is two nn.TransformerEncoderLayer
    with a key padding mask, which lose PyTorch's fused fast path when compiled and were about 50 ms
    slower on padded multi-row batches.
    """
    import copy

    import torch

    served = _served(agent)
    eager = served.eager
    whole = torch.compile(eager, dynamic=True)
    encoder_only = copy.copy(eager)  # same parameters and submodules ...
    encoder_only._modules = dict(eager._modules)  # ... except the encoder slot
    encoder_only._modules["encoder"] = torch.compile(eager.encoder, dynamic=True)
    served.paths = (whole, encoder_only)


def apply(agent: Any, *, fp16: bool, compile: bool) -> bool:
    """Apply the requested options to one agent. Does nothing and returns False for a model on the CPU."""
    if str(getattr(agent, "device", "")).startswith("cpu"):
        return False
    if fp16:
        use_fp16_weights(agent)
    if compile:
        compile_agent(agent)
    return True


def compiled_graphs() -> int:
    """Graphs torch.compile has produced in this process."""
    from torch._dynamo.utils import counters

    return int(counters["stats"]["unique_graphs"])
