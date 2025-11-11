'''
Hàm vẽ hình và so sánh dự đoán giữa pred và truth

Tham số:
bs: batch_size

Trả về:
3 hình ảnh và cặp truth, pred
'''

import numpy as np
import matplotlib.pyplot as plt
from dataloader import dataprocess
import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

def pred_and_plot(model, bs = 32, samples = 3):
    def unnormalize(img):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        img = img * std[:, None, None] + mean[:, None, None]
        return img.clamp(0, 1)
        
    device = torch.device('cuda:0')

    datamodule = dataprocess.DataModule(bs)
    datamodule.setup(stage = 'test')
    test_loader = datamodule.test_dataloader()

    random_idx = np.random.randint(0,len(test_loader.dataset),samples)
    best_model = model
    best_model.eval()
    with torch.inference_mode():
        for id in random_idx:
            plt.figure(figsize=(10,4))
            plt.imshow(unnormalize(test_loader.dataset[id][0]).permute(1,2,0))
            plt.axis(False)
            logits = best_model(test_loader.dataset[id][0].unsqueeze(0).to(device))
            pred = logits.argmax(1)
            plt.title(["thuc te: " + str(test_loader.dataset[id][1]), "du doan: " + str(pred.item())])
            plt.show()

def show_confusion_matrix(model, datamodule):
    device = torch.device('cuda:0')
    datamodule.setup(stage = 'test')
    test_loader = datamodule.test_dataloader()
    predictions = []
    model.eval()
    with torch.inference_mode():
        for X,_ in test_loader:
            logits = model(X.to(device))
            predictions.append(logits.argmax(1).to('cpu'))
        predictions = torch.concat(predictions)
    confmat = ConfusionMatrix(task = 'multiclass', num_classes=len(datamodule.get_class_names()))
    confmat_data = confmat(predictions, torch.Tensor(test_loader.dataset.targets))
    fig, ax = plot_confusion_matrix(conf_mat=confmat_data.numpy(),
                                    class_names=datamodule.get_class_names(),
                                    figsize=(10,5))
    plt.show()