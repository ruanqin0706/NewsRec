import os
import pickle
import time
from argparse import ArgumentParser

from sklearn.model_selection import StratifiedKFold
from torch.utils.tensorboard import SummaryWriter

from trainer import trainer
from utils import load_elmo_list, create_dir, load_elmo_list2
import numpy as np
import torch


def parse_args():
    parser = ArgumentParser()

    # gpu settings.
    parser.add_argument('--device_no', type=int)
    parser.add_argument('--seed', type=int)

    # data file path
    parser.add_argument('--labeled_elmo_path', nargs='+', default=[])
    parser.add_argument('--unlabeled_elmo_path', nargs='+', default=[])
    parser.add_argument('--category_size', type=int, default=-1)
    parser.add_argument('--max_len', type=int, default=200)
    parser.add_argument('--resume_teacher', action='store_true')
    parser.add_argument('--teacher_pre_trained', type=str)
    parser.add_argument('--resume_student', action='store_true')
    parser.add_argument('--student_pre_trained', type=str)

    # directory for saving files
    parser.add_argument('--board_dir', type=str)
    parser.add_argument('--teacher_dir', type=str)
    parser.add_argument('--other_dir', type=str)

    # training details
    parser.add_argument('--start_step', type=int)
    parser.add_argument('--total_steps', type=int)
    parser.add_argument('--labeled_bs', type=int)
    parser.add_argument('--unlabeled_bs', type=int)
    parser.add_argument('--lr_teacher', type=float)
    parser.add_argument('--lr_student', type=float)
    parser.add_argument('--use_strategy', action='store_true')
    parser.add_argument('--thr', type=str, default='static01',
                        choices=['static01', 'static46', 'tsa_exp', 'tsa_log', 'tsa_linear'])

    # info collections
    parser.add_argument('--c_train_prob', action='store_true')
    parser.add_argument('--c_val_prob', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # set flags / seeds
    seed = args.seed
    # torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    start_time = time.time()

    create_dir(dir_path=args.board_dir)
    create_dir(dir_path=args.other_dir)
    create_dir(dir_path=args.teacher_dir)
    writer = SummaryWriter(log_dir=args.board_dir)

    print(f'load embedding starting:')
    x_data, y_data = load_elmo_list(path_list=args.labeled_elmo_path, category_size=-1,
                                    max_len=args.max_len, shuffle=False)  # keep all, without shuffle
    print(f'load {len(y_data)} labeled data completed. {(time.time() - start_time) / 60} min.')
    x_over, y_over = load_elmo_list2(path_list=args.unlabeled_elmo_path, category_size=args.category_size,
                                     max_len=args.max_len,)
    print(f'load {len(y_over)} un-labeled data completed. {(time.time() - start_time) / 60} min.')

    score_dict_list = []
    # sk-learn provides 10-fold CV wrapper.
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    for i, (train_idx, valid_idx) in enumerate(kfold.split(x_data, y_data)):
        score_dict = trainer(args=args,
                             x_labeled=x_data[train_idx],
                             y_labeled=y_data[train_idx],
                             x_unlabeled=x_over,
                             y_unlabeled=y_over,
                             x_valid=x_data[valid_idx],
                             y_valid=y_data[valid_idx],
                             writer=writer,
                             fold_num=i)
        score_dict_list.append(score_dict)

    with open(os.path.join(args.other_dir, 'pickle.pkl'), 'wb') as f:
        pickle.dump(score_dict_list, f)
