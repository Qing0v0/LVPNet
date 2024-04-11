import os

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


class LossLog():
    def __init__(self, log_dir) -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir + '//loss')
        os.makedirs(self.log_dir + '//checkpoint')

        self.train_losses = []
        self.val_losses = []

    def append_loss(self, epoch, train_loss, val_loss):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        with open(os.path.join(self.log_dir, "loss\\train_loss.txt"), 'a') as f:
            f.write("epoch {}: {}\n".format(epoch, train_loss))
        with open(os.path.join(self.log_dir, "loss\\val_loss.txt"), 'a') as f:
            f.write("epoch {}: {}\n".format(epoch, val_loss))
        
        # plot losses
        plt.figure()
        plt.plot(range(len(self.train_losses)), self.train_losses, color="#6FBF9B", linewidth = 3, label='train loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, color="#E963A9", linewidth = 3, label='val loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "loss\\epoch_loss.png"))

        plt.cla()
        plt.close("all") 
    
    def get_log_dir(self):
        return self.log_dir
