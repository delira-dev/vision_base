from delira import get_backends as __get_backends

if "TORCH" in __get_backends():

    from .models.model_fns import create_resnet_torch, \
        create_vgg_torch, create_densenet_torch

    from .models.backbones import ResNetTorch, VGGTorch, AlexNetTorch, \
        SqueezeNetTorch, DenseNetTorch
