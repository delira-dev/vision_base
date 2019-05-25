import torch
from .nd_wrapper_torch import ConvWrapper, NormWrapper, PoolingWrapper, \
    DropoutWrapper
import typing
from torch import nn
import inspect
from copy import deepcopy


def _update_kwargs_to_correct_dim(kwargs: dict, dim):
    for key, value in kwargs.items():
        if isinstance(value, (tuple, list)):
            value = list(value)

            # add first value multiple times to ensure correct dimensionality;
            # won't have any effect if dim < len(value)
            value = [value[0]] * (dim - len(value)) + value

            # if dimension should be reduced: don't use all values; won't have
            # any effect if dim > ofiginal value length
            value = value[:dim]

            kwargs[key] = value

    return kwargs


def _convert_conv(conv: typing.Union[torch.nn.Conv1d,
                                     torch.nn.Conv2d,
                                     torch.nn.Conv3d,
                                     torch.nn.ConvTranspose1d,
                                     torch.nn.ConvTranspose2d,
                                     torch.nn.ConvTranspose3d],
                  dim: int,
                  stride=None):

    # accepted arguments won't change here by just changing the dimension
    kwargs = {key: getattr(conv, key)
              for key in inspect.signature(conv.__class__).parameters.keys()}

    kwargs = _update_kwargs_to_correct_dim(kwargs, dim)

    if stride is not None:
        # update strides to ensure correctness (needed e.g. if going from 2d to
        # 3d because we usually don't want cubic inputs)
        kwargs["stride"][0] = stride

    if kwargs["bias"] is not None:
        kwargs["bias"] = True
    else:
        kwargs["bias"] = False

    return ConvWrapper(dim, **kwargs)


def _convert_norm(norm: typing.Union[torch.nn.BatchNorm1d,
                                     torch.nn.BatchNorm2d,
                                     torch.nn.BatchNorm3d,
                                     torch.nn.GroupNorm,
                                     torch.nn.InstanceNorm1d,
                                     torch.nn.InstanceNorm2d,
                                     torch.nn.InstanceNorm3d,
                                     torch.nn.LayerNorm,
                                     torch.nn.LocalResponseNorm], dim):

    # accepted arguments won't change here by just changing the dimension
    kwargs = {key: getattr(norm, key)
              for key in inspect.signature(norm.__class__).parameters.keys()}

    # determine new norm type
    old_cls_name = norm.__class__.__name__

    # remove dimensionality
    if old_cls_name.endswith("d"):
        old_cls_name = old_cls_name[:-2]

    # class name is same for all dimensions -> don't pass any dimensionality to
    # wrapper
    else:
        dim = None

    new_cls_name = old_cls_name.replace("Norm", "")

    return NormWrapper(new_cls_name, dim, **kwargs)


def _convert_pool(pool: typing.Union[torch.nn.AdaptiveAvgPool1d,
                                     torch.nn.AdaptiveAvgPool2d,
                                     torch.nn.AdaptiveAvgPool3d,
                                     torch.nn.AdaptiveMaxPool1d,
                                     torch.nn.AdaptiveMaxPool2d,
                                     torch.nn.AdaptiveMaxPool3d,
                                     torch.nn.AvgPool1d,
                                     torch.nn.AvgPool2d,
                                     torch.nn.AvgPool3d,
                                     torch.nn.FractionalMaxPool2d,
                                     torch.nn.LPPool1d,
                                     torch.nn.LPPool2d,
                                     torch.nn.MaxPool1d,
                                     torch.nn.MaxPool2d,
                                     torch.nn.MaxPool3d],
                  dim, stride=None):

    # accepted arguments won't change here by just changing the dimension
    kwargs = {key: getattr(pool, key)
              for key in inspect.signature(pool.__class__).parameters.keys()}

    kwargs = _update_kwargs_to_correct_dim(kwargs, dim)

    # update given stride
    if stride is not None:

        # make stride a list with a value per dimension
        # (same behavior as if stride were int)
        if isinstance(kwargs["stride"], int):
            kwargs["stride"] = [kwargs["stride"]] * dim

        kwargs["stride"][0] = stride

    # determine new pool type
    old_cls_name = pool.__class__.__name__

    # remove dimensionality
    if old_cls_name.endswith("d"):
        old_cls_name = old_cls_name[:-2]

    # class name is same for all dimensions -> don't pass any dimensionality to
    # wrapper
    else:
        dim = None

    new_cls_name = old_cls_name.replace("Pool", "")

    return PoolingWrapper(new_cls_name, dim, **kwargs)


def _convert_dropout(dropout: typing.Union[torch.nn.Dropout,
                                           torch.nn.Dropout2d,
                                           torch.nn.Dropout3d,
                                           torch.nn.AlphaDropout,
                                           torch.nn.FeatureAlphaDropout],
                     dim):

    # accepted arguments won't change here by just changing the dimension
    kwargs = {key: getattr(dropout, key)
              for key in inspect.signature(dropout.__class__).parameters.keys()}

    # determine new pool type
    old_cls_name = dropout.__class__.__name__

    # remove dimensionality
    if old_cls_name.endswith("d"):
        old_cls_name = old_cls_name[:-2]

    form = old_cls_name.replace("Dropout", "")

    return DropoutWrapper(n_dim=dim, form=form , **kwargs)


def __check_is_conv(conv):
    return isinstance(conv, torch.nn.modules.conv._ConvNd)


def __check_is_norm(norm):
    # _InstanceNorm is also subclass of _BatchNorm
    return isinstance(norm, (torch.nn.modules.normalization._BatchNorm,
                             torch.nn.GroupNorm,
                             torch.nn.LayerNorm,
                             torch.nn.LocalResponseNorm))


def __check_is_pool(pool):
    return isinstance(pool, (torch.nn.modules.pooling._AdaptiveAvgPoolNd,
                             torch.nn.modules.pooling._AdaptiveMaxPoolNd,
                             torch.nn.modules.pooling._AvgPoolNd,
                             torch.nn.modules.pooling._LPPoolNd,
                             torch.nn.modules.pooling._MaxPoolNd,
                             torch.nn.FractionalMaxPool2d,
                             ))


def __check_is_dropout(dropout):
    return isinstance(dropout, torch.nn.modules.dropout._DropoutNd)


def _convert_single_module(module: torch.nn.Module,
                           namestr: typing.Union[str, None], dim,
                           conv_strides=None, pool_strides=None):

    # stop criterion for recursion
    if namestr is None:
        return module

    if "." in namestr:
        curr_name, further_name = namestr.split(".", 1)

    else:
        curr_name, further_name = namestr, None

    module_to_convert = getattr(module, curr_name)

    if __check_is_conv(module_to_convert):
        if isinstance(conv_strides, int) or conv_strides is None:
            converted_module = _convert_conv(module_to_convert, dim,
                                             conv_strides)
        else:
            converted_module = _convert_conv(module_to_convert, dim,
                                             conv_strides.pop(0))

    elif __check_is_norm(module_to_convert):
        converted_module = _convert_norm(module_to_convert, dim)

    elif __check_is_pool(module_to_convert):
        if isinstance(pool_strides, int) or pool_strides is None:
            converted_module = _convert_pool(module_to_convert, dim,
                                             pool_strides)
        else:
            converted_module = _convert_pool(module_to_convert, dim,
                                             pool_strides.pop(0))

    elif __check_is_dropout(module_to_convert):
        converted_module = _convert_dropout(module_to_convert, dim)

    else:
        converted_module = _convert_single_module(module_to_convert,
                                                  further_name, dim,
                                                  conv_strides, pool_strides)

    setattr(module, curr_name, converted_module)
    return module


def convert_network_dimension_(network: torch.nn.Module, new_dim,
                              conv_strides=None, pool_strides=None):

    for name, module in list(network.named_modules())[1:]:
        network = _convert_single_module(network, name, new_dim, conv_strides,
                                         pool_strides)

    return network


def convert_network_dimension(network: torch.nn.Module, new_dim,
                              conv_strides=None, pool_strides=None):
    network_copy = deepcopy(network)

    return convert_network_dimension_(network_copy, new_dim, conv_strides,
                                      pool_strides)