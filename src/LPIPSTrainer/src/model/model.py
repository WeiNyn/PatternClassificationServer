from typing import Tuple

import torch

import pytorch_lightning as pl

from lpips import LPIPS


class PatternModel(pl.LightningModule):
    """
    PatternModel PatternModel by pytorch lightning
    """
    def __init__(self, model, lr=0.1, step_size=30, gamma=0.1, beta=0.5):
        """
        __init__ Create PatternModel for training

        Args:
            model (lpips.LPIPS): lpips model to train
            lr (float, optional): learning rate. Defaults to 0.1.
            step_size (int, optional): step size for learning rate scheduler. Defaults to 30.
            gamma (float, optional): gamma for learning rate scheduler. Defaults to 0.1.
            beta (float, optional): beta for Adam optimizer. Defaults to 0.5.
        """
        
        super().__init__()
        self.model = model
        
        self.loss = torch.nn.MSELoss()
        
        self.step_size=step_size
        self.gamma=gamma
        self.beta=beta
        self.lr=lr

    def forward(self, img0, img1, label=None):
        output = self.model(img0, img1)
        if label is not None:
            loss = self.loss(output.flatten(), label)

            return loss, output
        
        return torch.tensor(0.0), output

    def training_step(self, batch, batch_idx):
        img0 = batch['img_1']
        img1 = batch['img_2']
        label = batch['label']
        
        loss, output = self.forward(img0, img1, label)
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        return dict(
            loss=loss,
            prediction=output.flatten(),
            label=label.flatten()
        )

    def validation_step(self, batch, batch_idx):
        img0 = batch['img_1']
        img1 = batch['img_2']
        label = batch['label']
        
        loss, _ = self.forward(img0, img1, label)
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        img0 = batch['img_1']
        img1 = batch['img_2']
        label = batch['label']
        
        loss, _ = self.forward(img0, img1, label)
        
        self.log('test_loss', loss, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outputs):
        predictions = []
        labels = []
        
        for output in outputs:
            for out_labels in output['label'].detach().cpu():
                labels.append(out_labels)
        for output in outputs:
            for out_predictions in output['prediction'].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels)
        predictions = torch.stack(predictions)
        
        acc = torch.sum(abs(labels - predictions) > 0.3)/len(labels)
        
        self.log('1 - accuracy', acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.step_size, gamma=self.gamma)

        return dict(
            optimizer=optimizer,
            lr_scheduler=dict(
                scheduler=scheduler,
                interval='epoch'
            )
        )


  
