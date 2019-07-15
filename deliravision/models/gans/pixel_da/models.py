import torch


class ResidualBlock(torch.nn.Module):
    def __init__(self, num_filts):
        super().__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(num_filts, num_filts, 3, 1, 1),
            torch.nn.BatchNorm2d(num_filts),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(num_filts, num_filts, 3, 1, 1),
            torch.nn.BatchNorm2d(num_filts),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(torch.nn.Module):
    def __init__(self, latent_dim, num_channels, img_size, n_residual_blocks,
                 num_filts=64):
        super().__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = torch.nn.Linear(latent_dim, num_channels * img_size ** 2)

        self.l1 = torch.nn.Sequential(
            torch.nn.Conv2d(num_channels * 2, 64, 3, 1, 1),
            torch.nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(n_residual_blocks):
            resblocks.append(ResidualBlock(num_filts))
        self.resblocks = torch.nn.Sequential(*resblocks)

        self.l2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, num_channels, 3, 1, 1), torch.nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Discriminator(torch.nn.Module):
    def __init__(self, num_channels):
        super().__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [torch.nn.Conv2d(in_features, out_features, 3, stride=2,
                                      padding=1),
                      torch.nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_features))
            return layers

        self.model = torch.nn.Sequential(
            *block(num_channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            torch.nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


class Classifier(torch.nn.Module):
    def __init__(self, num_channels, img_size, n_classes):
        super().__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [torch.nn.Conv2d(in_features, out_features, 3,
                                      stride=2, padding=1),
                      torch.nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(torch.nn.InstanceNorm2d(out_features))
            return layers

        self.model = torch.nn.Sequential(
            *block(num_channels, 64, normalization=False), *block(64, 128),
            *block(128, 256), *block(256, 512)
        )

        # downsampled size
        dsize = self.model(torch.rand(1, num_channels, img_size, img_size)
                           ).size(2)
        self.output_layer = torch.nn.Sequential(
            torch.nn.Linear(512 * dsize ** 2, n_classes),
            torch.nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label
