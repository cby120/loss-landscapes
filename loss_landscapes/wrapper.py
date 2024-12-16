import copy
import typing
import torch
from torch import nn
import numpy as np
from functools import partial
from tqdm import trange
import typing as t

from loss_landscapes.model_interface.model_wrapper import ModelWrapper, wrap_model
from loss_landscapes.model_interface.model_parameters import rand_u_like, orthogonal_to, ModelParameters
from loss_landscapes.metrics import Metric, Loss

t_Norm = t.Optional[t.Literal["model", "layer", "filter"]]


class LossSurface:
    def __init__(self, model: t.Union[nn.Module, ModelWrapper], loss_fn: t.Callable[[torch.Tensor], torch.Tensor],
                 span: float = 1e-1, steps: int = 20, norm: t_Norm = None, use_tqdm: bool = True) -> None:
        """
        Wrap a torch.nn.Module instance to calculate loss distribution in a 2d subspace of parameter space.
        2d subspace is randomly selected, with different methods of normalization to balance the scale of
        exploration on different parameters.

        Usage:
            model should load the intended state, e.g. trained parameters as the center point to explorer

            refer to loss_landscape.model_interface.model_wrapper if behavior of calling model.forward
                needs specifying, e.g. reset the model after each forward call;
                inherit one of the abstract classes and define the forward method
                or use GeneralModelWrapper and pass a forward_fn.
                if torch.nn.Module instance is directly passed in, SimpleModelWrapper will be applied.

            Referring to the usage of torch.nn.Module:
                to(device), load_state_dict(), train(), eval()

            use LossSurface().step(x, y) to calculate loss plane, return in np.ndarray(size=(steps, steps))
                (x, y) is the target pair from the dataset

        Changes:
            1. new way of specifying parameters, filter instead of passing sequence of modules
               (modules: list(torch.nn.Module) => model: torch.nn.Module)
            2. random_plane uses the same subspace by fixing dir1 and dir2 generated at instantiation;
               re-generate subspace by calling LossSurface by init_random_plane
            3. More precise way to iterate the grid points by anchor + shift instead of repeatedly operations

        :param model: torch.nn.Module
        :param loss_fn:
            evaluation function, e.g. loss functions like torch.nn.MSELoss
        :param span:
            relative range to explorer the parameter space, choose according to the interested scale of the
            structure in loss distribution
        :param steps:
            step size or grid interval to scan within the subspace, defining the finess of the output
        :param norm:
            methods to balance the effect of shift in different parameters
        :param use_tqdm:
            print progress bar duing scanning among the grids
        """
        model.eval()
        self.model = wrap_model(model)
        # self.model = model
        # [print(k) for k, v in self.model.named_parameters()]
        self.loss_fn = loss_fn
        self.dist = 1 / span
        self.steps = steps
        self.param_hook = self.model.get_module_parameters()
        self.center_param = self.param_hook.copy()
        self.dir1: t.Optional[ModelParameters] = None
        self.dir2: t.Optional[ModelParameters] = None
        self.dir1_norm: t.Optional[ModelParameters] = None
        self.dir2_norm: t.Optional[ModelParameters] = None
        self.init_random_plane()
        self.norm = norm
        if use_tqdm:
            self._range = partial(trange, ncols=140)
        else:
            self._range = range

    @property
    def norm(self):
        return self._norm

    @norm.setter
    def norm(self, norm: t_Norm):
        if norm not in ("model", "layer", "filter", None):
            raise ValueError(f"Unsupported normalization method: {norm}")
        self._norm = norm
        self.norm_plane()

    def load_state_dict(self, state_dict: typing.OrderedDict):
        self.model.load_state_dict(state_dict)
        self.center_param = self.param_hook.copy()  # update freezed center param
        self.norm_plane()

    def to(self, device: str):
        self.model.to(device)
        self.center_param.to(device)
        self.dir1.to(device)
        self.dir2.to(device)
        self.dir1_norm.to(device)
        self.dir2_norm.to(device)
        return self

    def init_random_plane(self):
        param_template = self.param_hook
        self.dir1 = rand_u_like(param_template)
        self.dir2 = orthogonal_to(self.dir1)

    def norm_plane(self):
        self.dir1_norm = self.dir1.copy()
        self.dir2_norm = self.dir2.copy()
        if self.norm == 'model':
            self.dir1_norm.model_normalize_(self.param_hook)
            self.dir2_norm.model_normalize_(self.param_hook)
        elif self.norm == 'layer':
            self.dir1_norm.layer_normalize_(self.param_hook)
            self.dir2_norm.layer_normalize_(self.param_hook)
        elif self.norm == 'filter':
            self.dir1_norm.filter_normalize_(self.param_hook)
            self.dir2_norm.filter_normalize_(self.param_hook)
        self.dir1_norm.mul_(((self.param_hook.model_norm() * self.dist) / self.steps) / self.dir1_norm.model_norm())
        self.dir2_norm.mul_(((self.param_hook.model_norm() * self.dist) / self.steps) / self.dir2_norm.model_norm())

    def random_plane(self, metric: Metric) -> np.ndarray:
        """
        Returns the computed value of the evaluation function applied to the model or agent along a planar
        subspace of the parameter space defined by a start point and two randomly sampled directions.
        The models supplied can be either torch.nn.Module models, or ModelWrapper objects
        from the loss_landscapes library for more complex cases.

        That is, given a neural network model, whose parameters define a point in parameter
        space, and a distance, the loss is computed at 'steps' * 'steps' points along the
        plane defined by the two random directions, from the start point up to the maximum
        distance in both directions.

        Note that the dimensionality of the model parameters has an impact on the expected
        length of a uniformly sampled other in parameter space. That is, the more parameters
        a model has, the longer the distance in the random other's direction should be,
        in order to see meaningful change in individual parameters. Normalizing the
        direction other according to the model's current parameter values, which is supported
        through the 'normalization' parameter, helps reduce the impact of the distance
        parameter. In future releases, the distance parameter will refer to the maximum change
        in an individual parameter, rather than the length of the random direction other.

        Note also that a simple planar approximation with randomly sampled directions can produce
        misleading approximations of the loss landscape due to the scale invariance of neural
        networks. The sharpness/flatness of minima or maxima is affected by the scale of the neural
        network weights. For more details, see `https://arxiv.org/abs/1712.09913v3`. It is
        recommended to normalize the directions, preferably with the 'filter' option.

        The Metric supplied has to be a subclass of the loss_landscapes.metrics.Metric class,
        and must specify a procedure whereby the model passed to it is evaluated on the
        task of interest, returning the resulting quantity (such as loss, loss gradient, etc).

        :param model: the model defining the origin point of the plane in parameter space
        :param metric: function of form evaluation_f(model), used to evaluate model loss
        :param distance: maximum distance in parameter space from the start point
        :param steps: at how many steps from start to end the model is evaluated
        :param normalization: normalization of direction vectors, must be one of 'filter', 'layer', 'model'
        :param deepcopy_model: indicates whether the method will deepcopy the model(s) to avoid aliasing
        :return: 1-d array of loss values along the line connecting start and end models
        """

        data_matrix = np.empty((self.steps, self.steps))
        coords = np.arange(-self.steps // 2, self.steps // 2)
        # alignment to original xy mapping
        # p2_arr, p1_arr = np.meshgrid(coords, coords)
        for i in self._range(self.steps):
            # i => dir_one, p1
            dir1_shift_base = self.center_param + self.dir1_norm * coords[i]
            for j in range(self.steps):
                self.param_hook.assign(dir1_shift_base + self.dir2_norm * coords[j])
                data_matrix[i, j] = metric(self.model)

        return data_matrix

    def step(self, x: torch.Tensor, y: torch.Tensor):

        loss = Loss(self.loss_fn, x, y)
        with torch.no_grad():
            plane = self.random_plane(loss)

        return plane

