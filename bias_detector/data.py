import numpy as np
from torch.utils.data import Dataset, DataLoader

from inference import predict_res
from utils import collate_func


class DataModule(Dataset):

    def __init__(self, x, y):
        # load elmo embedding file
        assert len(x) == len(y)
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def generate_loader(x_input, y_input, batch_size, device_no,
                    strategy=None, net=None, thr_start=0.0, thr_end=1.0):
    if strategy is None:
        data_set = DataModule(x=x_input, y=y_input)
    elif strategy.startwiths('thr_'):
        val_set = DataModule(x=x_input, y=y_input)
        val_loader = DataLoader(dataset=val_set, batch_size=batch_size,
                                collate_fn=lambda b: collate_func(b, device_no=device_no))
        y_prediction_arr = predict_res(net=net, data_loader=val_loader)
        keep_index = np.argwhere(
            np.logical_and(y_prediction_arr >= thr_start, y_prediction_arr <= thr_end)).reshape(-1)
        x_new, y_new = x_input[keep_index], y_input[keep_index]
        data_set = DataModule(x=x_new, y=y_new)
    else:
        1 / 0
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size,
                             collate_fn=lambda b: collate_func(b, device_no=device_no))
    return data_loader, data_set
