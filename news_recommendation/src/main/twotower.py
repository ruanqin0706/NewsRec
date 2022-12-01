import torch.multiprocessing as mp
from utils.manager import Manager
from models.TwoTower import TwoTowerModel
from torch.nn.parallel import DistributedDataParallel as DDP
from models.modules.encoder import *
import os
import pickle


def main(rank, manager):
    """ train/dev/test the model (in distributed)

    Args:
        rank: current process id
        world_size: total gpus
    """
    manager.setup(rank)
    loaders = manager.prepare()

    if manager.newsEncoder == "cnn":
        newsEncoder = CnnNewsEncoder(manager)
    elif manager.newsEncoder == "bert":
        newsEncoder = AllBertNewsEncoder(manager)
    elif manager.newsEncoder == "tfm":
        newsEncoder = TfmNewsEncoder(manager)
    if manager.userEncoder == "rnn":
        userEncoder = RnnUserEncoder(manager)
    elif manager.userEncoder == "sum":
        userEncoder = SumUserEncoder(manager)
    elif manager.userEncoder == "avg":
        userEncoder = AvgUserEncoder(manager)
    elif manager.userEncoder == "attn":
        userEncoder = AttnUserEncoder(manager)
    elif manager.userEncoder == "tfm":
        userEncoder = TfmUserEncoder(manager)

    model = TwoTowerModel(manager, newsEncoder, userEncoder).to(manager.device)

    if manager.mode == 'train':
        if manager.world_size > 1:
            model = DDP(model, device_ids=[manager.device], output_device=manager.device)
        manager.train(model, loaders)

    elif manager.mode == 'dev':
        manager.load(model)
        model.dev(manager, loaders, log=True)

    elif manager.mode == 'test':
        print(f'launch test evaluation on twotower')
        manager.load(model)
        model.eval()
        preds = model._test(manager, loaders)
        os.makedirs(manager.infer_dir, exist_ok=True)
        save_path = os.path.join(manager.infer_dir, f"{manager.suffix}_predictions.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(preds, f)
        print(f'save prediction results to {save_path}')


if __name__ == "__main__":
    config = {
        "enable_fields": ["title"],
        "newsEncoder": "cnn",
        "userEncoder": "rnn",
    }
    manager = Manager(config)

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)
