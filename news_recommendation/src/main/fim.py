import os
import pickle
import torch
import torch.multiprocessing as mp
from utils.manager import Manager
from models.FIM import FIM
from torch.nn.parallel import DistributedDataParallel as DDP


def main(rank, manager):
    manager.setup(rank)
    loaders = manager.prepare()

    model = FIM(manager).to(manager.device)

    if manager.mode == 'train':
        if manager.world_size > 1:
            model = DDP(model, device_ids=[manager.device], output_device=manager.device)
        manager.train(model, loaders)

    elif manager.mode == 'dev':
        manager.load(model)
        model.dev(manager, loaders, log=True)

    elif manager.mode == 'test':
        print(f'launch test evaluation')
        manager.load(model)
        model.eval()
        preds = model._test(manager, loaders)
        os.makedirs(manager.infer_dir, exist_ok=True)
        save_path = os.path.join(manager.infer_dir, f"{manager.suffix}_predictions.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(preds, f)
        print(f'save prediction results to {save_path}')
        # model.test(manager, loaders)


if __name__ == "__main__":
    config = {
        "batch_size": 100,
        "batch_size_eval": 100,
        "enable_fields": ["title"],
        "hidden_dim": 150,
        "learning_rate": 1e-5,
        "validate_step": "0.5e",
    }
    manager = Manager(config)

    # essential to set this to False to speed up dilated cnn
    torch.backends.cudnn.deterministic = False

    if manager.world_size > 1:
        mp.spawn(
            main,
            args=(manager,),
            nprocs=manager.world_size,
            join=True
        )
    else:
        main(manager.device, manager)
