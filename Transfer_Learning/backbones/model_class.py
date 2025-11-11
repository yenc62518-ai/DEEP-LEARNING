'''
Class khai báo model TinyVGG:

Tham số:
input: channel ảnh (int)
output_size: số lượng label (int)
kernel: số filter (int)
lr (float)

Cấu trúc: 
1 Conv2d (input, kernel, 3-same)
3 Conv2d (kernel, kernel, 3-same)
2 MaxPool2d (2)
Flatten()
Linear(kernel*H*W/4, output_size)

Trả về:
forward của cấu trúc
training_step() với CEL, f1, adam(lr)
test_step(), validation_step() với CEL, f1
'''
from torch import nn
import torch
from torchmetrics import F1Score
import pytorch_lightning as PL

class TinyVGG(PL.LightningModule) :
    def __init__(self, input, output_size, kernel, lr):
        super().__init__()
        self.save_hyperparameters()
        self.train_f1 = F1Score(task = 'multiclass', num_classes=output_size)
        self.val_f1 = F1Score(task = 'multiclass', num_classes=output_size)
        self.test_f1 = F1Score(task = 'multiclass', num_classes=output_size)
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input,
                      out_channels=kernel,
                      kernel_size=3,
                      padding = 'same',
                      stride = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=kernel,
                      out_channels=kernel,
                      kernel_size=3,
                      padding = 'same',
                      stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # 32, 16, 14, 14
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(kernel,
                      kernel,
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.Conv2d((kernel),
                      (kernel),
                      kernel_size=3,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2) # 32, 16, 7, 7
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear((kernel)*16*16, output_size)
        )
    def forward(self, X) :
        X = self.block_1(X)
        X = self.block_2(X)
        X = self.classification(X)
        return X
    
    def training_step(self, batch, batch_idx) :
        X,y = batch
        logits = self(X) # đang tự gọi hàm forward của nó
        pred = torch.argmax(logits, dim=1)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_f1.update(pred, y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_f1, prog_bar=True)
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
        self.log('test_acc', self.test_f1, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        X, label = batch
        return self(X)
    
    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9) 
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        
    