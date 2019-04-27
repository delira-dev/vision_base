from delira import get_backends as __get_backends

if "TORCH" in __get_backends():

    from .models.model_fns import resnet18_torch, resnet34_torch, \
        resnet50_torch, resnet101_torch, resnet152_torch, create_resnet_torch, \
        alexnet_torch, vgg11_torch, vgg13_torch, vgg16_torch, vgg19_torch, \
        create_vgg_torch, squeezenet1_0_torch, squeezenet1_1_torch, \
        densenet_121_torch, densenet_161_torch, densenet_169_torch, \
        densenet_201_torch, create_densenet_torch

    from .models.backbones import ResNetTorch, VGGTorch, AlexNetPyTorch, \
        SqueezeNetTorch, DenseNetPyTorch
