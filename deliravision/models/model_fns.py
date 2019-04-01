
from delira import get_backends

if "TORCH" in get_backends():
    from .backends.resnet import ResNetTorch as ResNet, \
        BasicBlockTorch as BasicBlock, BottleneckTorch as Bottleneck
    from .backends.vgg import VGGTorch as VGG
    from .backends.alexnet import AlexNetTorch as AlexNet

    RESNET_CONFIGS = {
        '18': {"block": BasicBlock, "layers": [2, 2, 2, 2]},
        "34": {"block": BasicBlock, "layers": [3, 4, 6, 3]},
        "50": {"block": Bottleneck, "layers": [3, 4, 6, 3]},
        "101": {"block": Bottleneck, "layers": [3, 4, 23, 3]},
        "152": {"block": Bottleneck, "layers": [3, 8, 36, 3]},
    }

    VGG_CONFIGS = {
        '11': [64, 'P', 128, 'P', 256, 256, 'P', 512, 512, 'P', 512, 512, 'P'],
        '13': [64, 64, 'P', 128, 128, 'P', 256, 256, 'P', 512, 512, 'P', 512,
               512, 'P'],
        '16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512,
               'P', 512, 512, 512, 'P'],
        '19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512,
               512, 512, 'P', 512, 512, 512, 512, 'P'],
    }

    def create_resnet_torch(num_layers: int, num_classes=1000, in_channels=3,
                            zero_init_residual=False, norm_layer=None, n_dim=2):
        config = RESNET_CONFIGS[str(num_layers)]

        return ResNet(config["block"], config["layers"],
                      num_classes=num_classes,
                      in_channels=in_channels,
                      zero_init_residual=zero_init_residual,
                      norm_layer=norm_layer, n_dim=n_dim)

    def create_vgg_torch(num_layers: int, num_classes=1000, in_channels=3,
                         init_weights=True, n_dim=2, norm_type="Batch",
                         pool_type="Max"):

        config = VGG_CONFIGS[str(num_layers)]

        return VGG(config, num_classes=num_classes,
                   in_channels=in_channels, init_weights=init_weights,
                   n_dim=n_dim, norm_type=norm_type, pool_type=pool_type)

    def resnet18_torch(num_classes=1000, in_channels=3,
                       zero_init_residual=False, norm_layer=None, n_dim=2):
        """Constructs a ResNet-18 model.
        """
        return create_resnet_torch(18, num_classes=num_classes,
                                   in_channels=in_channels,
                                   zero_init_residual=zero_init_residual,
                                   norm_layer=norm_layer, n_dim=n_dim)

    def resnet34_torch(num_classes=1000, in_channels=3,
                       zero_init_residual=False, norm_layer=None, n_dim=2):
        """Constructs a ResNet-34 model.
        """
        return create_resnet_torch(34, num_classes=num_classes,
                                   in_channels=in_channels,
                                   zero_init_residual=zero_init_residual,
                                   norm_layer=norm_layer, n_dim=n_dim)


    def resnet50_torch(num_classes=1000, in_channels=3,
                       zero_init_residual=False, norm_layer=None, n_dim=2):
        """Constructs a ResNet-50 model.
        """
        return create_resnet_torch(50, num_classes=num_classes,
                                   in_channels=in_channels,
                                   zero_init_residual=zero_init_residual,
                                   norm_layer=norm_layer, n_dim=n_dim)


    def resnet101_torch(num_classes=1000, in_channels=3,
                        zero_init_residual=False, norm_layer=None, n_dim=2):
        """Constructs a ResNet-101 model.
        """
        return create_resnet_torch(101, num_classes=num_classes,
                                   in_channels=in_channels,
                                   zero_init_residual=zero_init_residual,
                                   norm_layer=norm_layer, n_dim=n_dim)

    def resnet152_torch(num_classes=1000, in_channels=3,
                        zero_init_residual=False, norm_layer=None, n_dim=2):
        """Constructs a ResNet-152 model.
        """
        return create_resnet_torch(152, num_classes=num_classes,
                                   in_channels=in_channels,
                                   zero_init_residual=zero_init_residual,
                                   norm_layer=norm_layer, n_dim=n_dim)


    def vgg11(num_classes=1000, in_channels=3, init_weights=True, n_dim=2,
              norm_type="Batch", pool_type="Max"):

        return create_vgg_torch(11, num_classes=num_classes,
                                in_channels=in_channels,
                                init_weights=init_weights, n_dim=n_dim,
                                norm_type=norm_type, pool_type=pool_type)

    def vgg13(num_classes=1000, in_channels=3, init_weights=True, n_dim=2,
              norm_type="Batch", pool_type="Max"):

        return create_vgg_torch(13, num_classes=num_classes,
                                in_channels=in_channels,
                                init_weights=init_weights, n_dim=n_dim,
                                norm_type=norm_type, pool_type=pool_type)

    def vgg16(num_classes=1000, in_channels=3, init_weights=True, n_dim=2,
              norm_type="Batch", pool_type="Max"):

        return create_vgg_torch(16, num_classes=num_classes,
                                in_channels=in_channels,
                                init_weights=init_weights, n_dim=n_dim,
                                norm_type=norm_type, pool_type=pool_type)

    def vgg19(num_classes=1000, in_channels=3, init_weights=True, n_dim=2,
              norm_type="Batch", pool_type="Max"):

        return create_vgg_torch(19, num_classes=num_classes,
                                in_channels=in_channels,
                                init_weights=init_weights, n_dim=n_dim,
                                norm_type=norm_type, pool_type=pool_type)

    def alexnet(num_classes=1000, in_channels=3, n_dim=2,
                pool_type="Max"):
        return AlexNet(num_classes=num_classes, in_channels=in_channels,
                       n_dim=n_dim, pool_type=pool_type)
