""" Class used to define interface to complex models """

import abc
import itertools
from torch import nn
import typing as t

from ..model_interface.model_parameters import ModelParameters

Self = t.TypeVar("Self")
Param_Mod_t = t.Union[nn.Module, str, t.Type, nn.Parameter]
Filter_t = t.Optional[t.Sequence[Param_Mod_t]]


class ModelWrapper(abc.ABC):
    def __init__(self, model: nn.Module):
        self.model = model
        self.tracked_params = [p for p in self.model.parameters()]

    def get_module_parameters(self) -> ModelParameters:
        return ModelParameters(self.tracked_params)

    def train(self: Self, mode=True) -> Self:
        self.model.train(mode)

    def eval(self) -> Self:
        return self.model.eval()

    def to(self, device: str) -> Self:
        self.model = self.model.to(device, non_blocking=True)

    def requires_grad_(self, requires_grad=True) -> Self:
        for p in self.model.parameters():
            p.requires_grad = requires_grad
        return self

    def zero_grad(self, set_to_none: bool = False) -> Self:
        self.model.zero_grad(set_to_none)
        return self

    def parameters(self, recurse: bool = True) -> t.Iterator[nn.Parameter]:
        return self.model.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> t.Iterator[t.Tuple[str, nn.Parameter]]:
        return self.model.named_parameters(prefix, recurse)

    @abc.abstractmethod
    def forward(self, x):
        pass


class SimpleModelWrapper(ModelWrapper):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def forward(self, x):
        return self.model(x)


class FilteredModelWrapper(ModelWrapper):
    def __init__(self, model: nn.Module, select: Filter_t = None, drop: Filter_t = None):
        """
        Manually specify module / module name / module type / parameter instance to explorer
        in torch.nn.Module during generating loss landscape

        :param model: torch.nn.Module
        :param select, drop: optional
            sequence of (str, torch.nn.Module, type, torch.nn.Parameter)
            select for white-list, drop for black list in explorering params
            drop will take no effect if select is not None
        :raises ValueError: if select and drop have no intersection
        """
        super().__init__(model)
        select = set(select) if select is not None else None
        drop = set(drop) if drop is not None else None
        if drop is not None and select is not None:
            if select.intersection(drop):  # ensure white list and black list have no intersection
                raise ValueError("select and drop filter have overlap elements")
        self.select = select
        self.drop = drop
        self.tracked_params = {k: v for k, v in self.model.named_parameters() if self.in_filter(k)}

    def in_filter(self, p_name: str):
        p_module = self.model.get_submodule(".".join(p_name.split(".")[:-1]))
        p_type = type(p_module)

        if self.select:  # white list
            if p_type in self.select or p_type.__name__ in self.select:
                return True
            for mod in self.select:  # torch.nn.Module.modules() contains itself
                if not isinstance(mod, nn.Module):
                    continue
                if p_module in mod.modules():
                    return True
            for name in self.select:
                if not isinstance(name, str):
                    continue
                if name in p_name:  # name is substring of param name
                    return True
            return False

        if self.drop:  # black list
            if p_type in self.drop or p_type.__name__ in self.drop:
                return False
            for mod in self.drop:  # torch.nn.Module.modules() contains itself
                if not isinstance(mod, nn.Module):
                    continue
                if p_module in mod.modules():
                    return False
            for name in self.drop:
                if not isinstance(name, str):
                    continue
                if name in p_name:  # name is substring of param name
                    return False

        return True

    def parameters(self, recurse: bool = True) -> t.Iterator[nn.Parameter]:
        return iter(p for n, p in self.model.named_parameters(recurse) if n in self.tracked_params)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> t.Iterator[t.Tuple[str, nn.Parameter]]:
        return iter((n, p) for n, p in self.model.named_parameters(prefix, recurse) if n in self.tracked_params)


class GeneralModelWrapper(ModelWrapper):
    def __init__(self, model, modules: list, forward_fn):
        super().__init__(model)
        self.modules = modules
        self.forward_fn = forward_fn

    def forward(self, x):
        return self.forward_fn(self.model, x)


def wrap_model(model) -> ModelWrapper:
    if isinstance(model, ModelWrapper):
        return model.requires_grad_(False)
    elif isinstance(model, nn.Module):
        return SimpleModelWrapper(model).requires_grad_(False)
    else:
        raise ValueError('Only models of type torch.nn.modules.module.Module can be passed without a wrapper.')
