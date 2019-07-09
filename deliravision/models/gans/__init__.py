from deliravision.models.gans.adversarial_autoencoder import *
from deliravision.models.gans.auxiliary_classifier import *
from deliravision.models.gans.boundary_equilibrium import *
from deliravision.models.gans.boundary_seeking import *
from deliravision.models.gans.conditional import *
from deliravision.models.gans.context_conditional import *
from deliravision.models.gans.context_encoder import *
from deliravision.models.gans.coupled import *
from deliravision.models.gans.cycle import *
from deliravision.models.gans.deep_convolutional import *
from deliravision.models.gans.disco import *
from deliravision.models.gans.dragan import *
from deliravision.models.gans.dual import *
from deliravision.models.gans.energy_based import *
from deliravision.models.gans.esr import *
from deliravision.models.gans.gan import *

# make LSGAN a synonym for basic GAN, since training only differs in loss
# function, which isn't specified here
LeastSquareGAN = GenerativeAdversarialNetworks
