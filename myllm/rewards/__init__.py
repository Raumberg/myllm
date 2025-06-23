from __future__ import annotations

"""Reward functions/classes registry.

Ported from v1 – each reward is exposed as a *callable* returning a list[float]
for a batch. For ultimate flexibility we allow two styles:

1. Function-based reward  → wrap with `FunctionReward`.
2. Class-based reward inheriting from `BaseReward` with `__call__`.

The public helper `get_reward(name)` resolves by key.
"""

from importlib import import_module
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Sequence

__all__ = [
    "BaseReward",
    "WeightedReward",
    "register_reward",
    "get_reward",
    "build_rewards",
]


# ------------------------------------------------------------------
# Core interfaces
# ------------------------------------------------------------------


class BaseReward:  # noqa: D401
    """Abstract base for reward callables.

    Subclasses must set ``name`` and implement ``__call__(batch)`` returning
    list[float] of length ``len(batch)``.
    """

    name: str

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        raise NotImplementedError

    def __repr__(self) -> str:  # noqa: D401
        return f"{self.__class__.__name__}()"


class WeightedReward(BaseReward):
    """Wrapper applying scalar weight to inner reward list."""

    def __init__(self, reward: BaseReward, weight: float = 1.0):
        self.reward = reward
        self.weight = weight
        self.name = reward.name
        # Some external libraries expect a callable to expose `__name__` (e.g. TRL GRPOTrainer).
        # Provide it so that `inspect.getsource` & logging work without errors.
        self.__name__ = self.name

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        out = self.reward(*args, **kwargs)
        return [v * self.weight for v in out]

    def __repr__(self) -> str:  # noqa: D401
        return f"WeightedReward({self.reward}, weight={self.weight})"


class FunctionReward(BaseReward):
    """Adapter: arbitrary function -> BaseReward."""

    def __init__(self, func: Callable):
        self.func = func
        self.name = func.__name__

    def __call__(self, *args: Any, **kwargs: Any):  # noqa: D401
        return self.func(*args, **kwargs)


# ------------------------------------------------------------------
# Registry helpers
# ------------------------------------------------------------------


_REGISTRY: Dict[str, Callable[..., BaseReward]] = {}


def register_reward(cls):  # noqa: D401
    """Class decorator to register reward classes by ``cls.name``."""

    if not issubclass(cls, BaseReward):
        raise TypeError("@register_reward expects subclass of BaseReward")
    _REGISTRY[cls.name] = cls
    return cls


def list_rewards() -> Sequence[str]:  # noqa: D401
    return sorted(_REGISTRY)


def get_reward(name: str, *, weight: float | None = None, **kwargs) -> BaseReward:  # noqa: D401
    """Return instantiated reward by *name* with optional weight & params."""

    if name not in _REGISTRY:
        raise KeyError(f"Unknown reward '{name}'. Available: {list_rewards()}")

    reward_cls = _REGISTRY[name]
    reward = reward_cls(**kwargs)
    # Ensure each reward object exposes `__name__` as expected by TRL.
    if not hasattr(reward, "__name__"):
        setattr(reward, "__name__", reward.name)

    return WeightedReward(reward, weight) if weight is not None else reward


# ------------------------------------------------------------------
# Builder from config specs
# ------------------------------------------------------------------


def _parse_spec(spec: Any):  # noqa: D401
    """Parse single spec into (name, weight, params)."""

    if isinstance(spec, str):
        # Accept forms: "name", "name:0.5", "name@0.5"
        if ":" in spec:
            name, w = spec.split(":", 1)
            return name, float(w), {}
        if "@" in spec:
            name, w = spec.split("@", 1)
            return name, float(w), {}
        return spec, None, {}

    if isinstance(spec, dict):
        if len(spec) != 1:
            raise ValueError("Reward dict spec must have exactly one key")
        name, params = next(iter(spec.items()))
        weight = params.pop("weight", None) if isinstance(params, dict) else None
        return name, weight, params if isinstance(params, dict) else {}

    raise TypeError(f"Unsupported reward spec type: {type(spec)}")


def build_rewards(specs: Iterable[Any] | None) -> List[BaseReward]:  # noqa: D401
    """Build list of reward objects from iterable specs.

    If *specs* is None/empty → single LengthPenalty as default.
    """

    if not specs:
        specs = ["length_penalty"]

    rewards: List[BaseReward] = []
    for spec in specs:
        name, weight, params = _parse_spec(spec)
        rewards.append(get_reward(name, weight=weight, **params))

    return rewards


# ensure classes in submodules register themselves
from importlib import import_module as _imp

for _sub in ("textual", "math", "similarity", "correctness", "debug"):
    try:
        _imp(f"myllm.rewards.{_sub}")
    except ModuleNotFoundError:
        pass 