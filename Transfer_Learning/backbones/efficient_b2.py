'''
Class Transfered_Model được chỉnh trainable params và output_size 

Tham số:
output_size: số class muốn dự đoán
pretrained_model
lr = 0.003
unfreeze_lr: learning rate của unfreezed layer
'''
import torch 
from torch import nn
import pytorch_lightning as PL
import torchvision
from torchmetrics import F1Score
from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights

class Transfered_Model(PL.LightningModule):
    def __init__(self, output_size = 3, lr = 0.003, unfreeze_lr = None):
        super().__init__()
        self.save_hyperparameters()
        pretrained_model = efficientnet_b2(weights = EfficientNet_B2_Weights.DEFAULT)
        for params in pretrained_model.parameters():
            params.requires_grad = False
            
        self.train_f1 = F1Score(task = 'multiclass', num_classes=output_size)
        self.val_f1 = F1Score(task = 'multiclass', num_classes=output_size)
        self.test_f1 = F1Score(task = 'multiclass', num_classes=output_size)

        self.block_1 = pretrained_model.features
        self.block_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(1))
        self.block_3 = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(1408,output_size)
        )
    def unfreeze(self,num_trainable):
        params = list(self.block_1.parameters())
        for param in params[-num_trainable:]:
            param.requires_grad = True
        
    def forward(self, X):
        X = self.block_1(X)
        X = self.block_2(X)
        return self.block_3(X)
    def training_step(self, batch, batch_idx) :
        X,y = batch
        logits = self(X) # đang tự gọi hàm forward của nó
        pred = torch.argmax(logits, dim=1)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_f1.update(pred, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_f1.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_f1", self.val_f1, prog_bar=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        logits = self(X) # đang tự gọi hàm forward của nó
        pred = torch.argmax(logits, dim = 1)
        loss = nn.functional.cross_entropy(logits, y)
        self.test_f1.update(pred, y)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        X, label = batch
        return self(X)
    
    def configure_optimizers(self): 
        if self.hparams.unfreeze_lr is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) 
        else:
            optimizer = torch.optim.Adam([
                {'params': [p for p in self.block_1.parameters() if p.requires_grad], 'lr': self.hparams.unfreeze_lr},
                {'params': self.block_3.parameters(), 'lr': self.hparams.lr}
            ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7) 
        return {"optimizer": optimizer, "lr_scheduler": scheduler}