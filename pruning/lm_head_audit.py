"""Measure the retained Action Expert head and observe its use during sampling.

Tensor bytes are identity-deduplicated ``numel * element_size``, not measured
allocator/VRAM usage. Shared parameter identities are reported separately so a
head's logical size is not mistaken for memory that could be freed by removal.
"""

from contextlib import contextmanager

from torch import nn


HEAD_PATH = "paligemma_with_expert.gemma_expert.lm_head"


def _get_head(model):
    module = model
    for component in HEAD_PATH.split("."):
        module = getattr(module, component, None)
        if module is None:
            return None
    return module if isinstance(module, nn.Module) else None


def _count(parameters):
    result = {"parameters": 0, "weight_bytes": 0, "by_dtype": {}}
    seen = set()
    for parameter in parameters:
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        elements = parameter.numel()
        size = elements * parameter.element_size()
        result["parameters"] += elements
        result["weight_bytes"] += size
        dtype = result["by_dtype"].setdefault(str(parameter.dtype), {"parameters": 0, "weight_bytes": 0})
        dtype["parameters"] += elements
        dtype["weight_bytes"] += size
    result["weight_mib"] = result["weight_bytes"] / 1024**2
    return result


def lm_head_metrics(model: nn.Module) -> dict:
    """Count the actual head without changing it or allocating model copies.

    ``exclusive`` excludes parameters also registered outside this head. These are
    logical tensor bytes; distinct Parameters sharing storage are not deduplicated.
    This metric excludes activations, buffers, caches and allocator overhead.
    """
    head = _get_head(model)
    if head is None:
        return {"status": "not_present", "path": HEAD_PATH, "retained": False}
    outside_ids = {
        id(parameter)
        for name, parameter in model.named_parameters(remove_duplicate=False)
        if not name.startswith(HEAD_PATH + ".")
    }
    parameters = list(head.parameters())
    return {
        "status": "measured",
        "path": HEAD_PATH,
        "retained": True,
        **_count(parameters),
        "shared_with_rest": _count(parameter for parameter in parameters if id(parameter) in outside_ids),
        "exclusive": _count(parameter for parameter in parameters if id(parameter) not in outside_ids),
    }


@contextmanager
def observe_lm_head_calls(model: nn.Module):
    """Observe module invocations in one enclosed ``sample_actions`` call.

    The caller must enclose sampling only. The hook is removed even on failure; a
    failed/incomplete call remains unverified. A successful zero count describes
    only the observed path, not all possible inference methods. Direct calls to
    ``head.forward`` bypass PyTorch hooks and are outside this observation's scope.
    """
    head = _get_head(model)
    observation = {"status": "unverified", "forward_calls": 0}
    if head is None:
        observation.update(status="not_present", forward_calls=None)
        yield observation
        return

    def count_call(_module, _inputs):
        observation["forward_calls"] += 1

    handle = head.register_forward_pre_hook(count_call)
    try:
        yield observation
    except Exception as exc:
        observation["error"] = f"{type(exc).__name__}: {exc}"
        raise
    else:
        observation["status"] = (
            "called_on_observed_sample_actions"
            if observation["forward_calls"]
            else "not_called_on_observed_sample_actions"
        )
    finally:
        handle.remove()
