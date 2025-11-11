from backbones import efficient_b0, efficient_b2
from dataloader import dataprocess
import logger
import argparse
from torchvision.models import efficientnet_b0, efficientnet_b2
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b2

MAX_EPOCHS = 20
INPUT_SIZE = 3
NUM_CLASS = 3
KERNEL = 10
LEARNING_RATE = 0.00148
BATCH_SIZE = 23

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo argparse")
    parser.add_argument('--epochs', type=int, default=10, help='Số lần huấn luyện')
    parser.add_argument('--lr', type=float, default=0.001, help='Tốc độ học')
    parser.add_argument('--bs', type = int, default=36, help = 'batch size')
    args = parser.parse_args()
    
    trainer = logger.trainer(experiment='unfreeze_with_b2_high_lr',MAX_EPOCHS= args.epochs, stage = 'fit')
    model = efficient_b2.Transfered_Model(output_size=3, lr = args.lr, unfreeze_lr=args.lr*0.1)
    datamodule = dataprocess.DataModule(batch_size=args.bs, num_workers=2, persistent_worker = True)
    #model.lr = lr_find = logger.find_lr(model=model, datamodule=datamodule, trainer=trainer)
    trainer.fit(model, datamodule=datamodule)
    