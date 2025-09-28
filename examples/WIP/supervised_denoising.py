from deepinv.datasets.patch_dataset import AlternativePatchDataset3D
from deepinv.physics.noise import GaussianNoise
from deepinv.training import Trainer

import os

if __name__ == '__main__':
    dataset = AlternativePatchDataset3D(os.environ['IXI_T1'])
    physics = GaussianNoise(sigma=0.2)