import deepinv as dinv

from deepinv.datasets.patch_dataset import AlternativePatchDataset3D
from deepinv.physics.noise import GaussianNoise, RicianNoise
from deepinv.training import Trainer



import os
import torch


if __name__ == '__main__':
    dataset = AlternativePatchDataset3D(os.environ['IXI_T1'])
    physics = dinv.physics.forward.Physics(noise_model=RicianNoise(sigma=0.1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    model = dinv.models.dncnn.DnCNN(in_channels=1,
                                    out_channels=1,
                                    depth=4,
                                    nf=16,
                                    pretrained=None,
                                    device='cuda:0',
                                    dim=3)

    trainer = Trainer(
        model=model,
        physics=physics,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        train_dataloader=dataloader,
        epochs=50,
        losses=dinv.loss.SupLoss(metric=dinv.metric.MSE()),
        metrics=dinv.metric.PSNR(),
        device='cuda:0',
        show_progress_bar=True,
        log_train_batch=True,
        online_measurements=True
    )

    trainer.train()