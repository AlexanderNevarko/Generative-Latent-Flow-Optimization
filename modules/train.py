import torch
import torch.nn as nn
import pytorch_lightning as pl


class GLOOptimizer(torch.optim.Adam):
    def __init__(self, *args, z_param, zlr):
        super(GLOOptimizer, self).__init__(*args)
        self.z_param = z_param
        self.zlr = zlr
    
    def step(self, closure=None, z_grad=None):
        loss = super().step()
        
        self.z_param -= self.zlr * z_grad
        self.z_param /= torch.max(torch.tensor([torch.sqrt(torch.sum(self.z_param**2)), 1.0]))
        return loss
        


class GLO_pl(pl.LightningModule):
    
    def __init__(self, model, loss, zlr, experiment, exp_name):
        super(GLO_pl, self).__init__()
        self.model = model
        self.loss = loss
        self.zlr = zlr
        self.experiment = experiment
        self.experiment.set_name(exp_name)
        
    def training_step(self, batch, bacth_idx):
        z, img = batch
        pred = self.model(z)
        loss = self.loss(pred, img)
        self.experiment.log_metric()
        return loss
    
    def validation_step(self, batch, bacth_idx):
        z, img = batch.batch
        pred = self.model(z)
        loss = self.loss(pred, img)
        
    
    def configure_optimizers(self):
        return GLOOptimizer(self.model.parameters(), 
                            z_param=self.model.z, 
                            zlr=self.zlr)