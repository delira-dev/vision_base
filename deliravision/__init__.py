from delira import get_backends as __get_backends

if "TORCH" in __get_backends():

    from .models.model_fns import resnet18_torch, resnet34_torch, \
        resnet50_torch, resnet101_torch, resnet152_torch, create_resnet_torch, \
        alexnet, vgg11, vgg13, vgg16, vgg19, create_vgg_torch

    from .models.backends import ResNetTorch, VGGTorch, AlexNetTorch
