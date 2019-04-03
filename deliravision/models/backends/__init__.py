from delira import  get_backends as __get_backends

if "TORCH" in __get_backends():

    from .resnet import ResNetTorch
    from .vgg import VGGTorch
    from .alexnet import AlexNetTorch
    from .squeezenet import SqueezeNetTorch
    from .densenet import DenseNetTorch
