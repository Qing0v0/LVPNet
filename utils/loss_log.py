import os

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter


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
        plt.plot(range(len(self.train_losses)), self.train_losses, 'red', linewidth = 2, label='train loss')
        plt.plot(range(len(self.val_losses)), self.val_losses, 'coral', linewidth = 2, label='val loss')

        # plot smoothing losses
        if len(self.train_losses) > 5:
            window_length = 15 if len(self.train_losses) > 25 else 5
            plt.plot(
                range(len(self.train_losses)), 
                savgol_filter(self.train_losses, window_length, 3), 
                'green', linestyle ='--', linewidth=2, label='smoothing train loss'
            )
            plt.plot(
                range(len(self.val_losses)), 
                savgol_filter(self.val_losses, window_length, 3), 
                '#8B4513', linestyle ='--', linewidth=2, label='smoothing val loss'
            )

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "loss\\epoch_loss.png"))

        plt.cla()
        plt.close("all") 
    
    def get_log_dir(self):
        return self.log_dir
