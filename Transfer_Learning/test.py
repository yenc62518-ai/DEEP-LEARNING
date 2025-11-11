from backbones import efficient_b0
from dataloader import dataprocess
import logger
from model_checker import pred_and_plot, show_confusion_matrix
from torchvision.models import EfficientNet_B0_Weights

if __name__ == '__main__':
    weight = EfficientNet_B0_Weights.DEFAULT
    checkpoint = 'tb_log/unfreeze_with_higher_lr/lightning_logs/version_0/checkpoints/epoch=7-step=48.ckpt'
    best_model = efficient_b0.Transfered_Model.load_from_checkpoint(checkpoint_path=checkpoint)
    datamodule = dataprocess.DataModule(batch_size=32, num_workers=2)
    trainer = logger.trainer(experiment='test b0 high lr', stage = 'test')

    #trainer.test(model = best_model, datamodule=datamodule)
    show_confusion_matrix(model=best_model,datamodule=datamodule)
    pred_and_plot(model=best_model, bs = 16, samples=3)

