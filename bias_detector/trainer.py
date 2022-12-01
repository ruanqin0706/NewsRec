import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import generate_loader, DataModule
from inference import valid_performance, predict_res
from model import DetectionModel, weights_init
from utils import collate_func


def trainer(args, x_labeled, y_labeled, x_unlabeled, y_unlabeled, x_valid, y_valid, writer, fold_num):
    y_val_list = []
    y_train_list = []

    # instantiate / load network
    teacher_net = DetectionModel()
    student_net = DetectionModel()
    if args.resume_teacher:
        teacher_net.load_state_dict(torch.load(args.teacher_pre_trained))
        print(f'load pre-trained {args.teacher_pre_trained} successfully !')
    else:
        teacher_net.apply(weights_init)
    if args.device_no > -1:
        teacher_net = teacher_net.cuda(args.device_no)
    if args.resume_student:
        student_net.load_state_dict(torch.load(args.student_pre_trained))
        print(f'load pre-trained {args.student_pre_trained} successfully !')
    else:
        student_net.apply(weights_init)
    if args.device_no > -1:
        student_net = student_net.cuda(args.device_no)

    if args.c_val_prob:
        val_set = DataModule(x=x_valid, y=y_valid)
        val_loader = DataLoader(dataset=val_set, batch_size=args.labeled_bs,
                                collate_fn=lambda b: collate_func(b, device_no=args.device_no))
        y_arr = predict_res(net=teacher_net, data_loader=val_loader)
        y_val_list.append(y_arr)
    if args.c_train_prob:
        temp_set = DataModule(x=x_labeled, y=y_labeled)
        temp_loader = DataLoader(dataset=temp_set, batch_size=args.labeled_bs,
                                 collate_fn=lambda b: collate_func(b, device_no=args.device_no))
        y_temp = predict_res(net=teacher_net, data_loader=temp_loader)
        y_train_list.append(y_temp)

    teacher_net.train()
    student_net.train()

    # create losses (criterion in pytorch)
    criterion = nn.BCEWithLogitsLoss()

    # create optimizers
    teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=args.lr_teacher)
    student_optim = torch.optim.Adam(student_net.parameters(), lr=args.lr_student)

    # variables
    moving_dot_product = torch.empty(1)
    limit = 3.0 ** 0.5  # 3 = 6 / (f_in + f_out)
    nn.init.uniform_(moving_dot_product, -limit, limit)
    if args.device_no > -1:
        moving_dot_product = moving_dot_product.to(args.device_no)

    # first time load data
    labeled_loader, labeled_set = generate_loader(x_input=x_labeled, y_input=y_labeled, batch_size=args.labeled_bs,
                                                  device_no=args.device_no, strategy=None)
    unlabeled_loader, unlabeled_set = generate_loader(x_input=x_unlabeled, y_input=y_unlabeled,
                                                      batch_size=args.unlabeled_bs,
                                                      device_no=args.device_no, strategy=None)
    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)
    valid_set = DataModule(x=x_valid, y=y_valid)

    # record variables
    score_dict = {
        'f1_mode_f1': 0.0, 'f1_mode_acc': 0.0, 'f1_mode_recall': 0.0, 'f1_mode_precision': 0.0, 'f1_path': '',
        'acc_mode_f1': 0.0, 'acc_mode_acc': 0.0, 'acc_mode_recall': 0.0, 'acc_mode_precision': 0.0, 'acc_path': '',
    }

    # training strategies controlled.
    for step in range(args.start_step, args.total_steps + 1):
        try:
            x_l, y_l = labeled_iter.next()
        except:
            t_val_acc, t_val_recall, t_val_precision, t_val_f1 = valid_performance(net=teacher_net,
                                                                                   data_set=valid_set,
                                                                                   batch_size=args.labeled_bs,
                                                                                   device_no=args.device_no)
            if t_val_f1 > score_dict['f1_mode_f1']:
                model_file_path = os.path.join(args.teacher_dir,
                                               f'BNCNN_vf1{t_val_f1:.4f}_f{fold_num}_s{step:04d}_e{step // args.labeled_bs}.pth')
                torch.save(teacher_net.state_dict(), model_file_path)
                # best_acc = t_val_acc
                score_dict['f1_mode_f1'] = t_val_f1
                score_dict['f1_mode_acc'] = t_val_acc
                score_dict['f1_mode_recall'] = t_val_recall
                score_dict['f1_mode_precision'] = t_val_precision
                score_dict['f1_path'] = model_file_path
                print(f'save model at step: {step} epoch: {step // args.labeled_bs} based on f1.')
            if t_val_acc > score_dict['acc_mode_acc']:
                model_file_path = os.path.join(args.teacher_dir,
                                               f'BNCNN_vacc{t_val_acc:.4f}_f{fold_num}_s{step:04d}_e{step // args.labeled_bs}.pth')
                torch.save(teacher_net.state_dict(), model_file_path)
                score_dict['acc_mode_f1'] = t_val_f1
                score_dict['acc_mode_acc'] = t_val_acc
                score_dict['acc_mode_recall'] = t_val_recall
                score_dict['acc_mode_precision'] = t_val_precision
                score_dict['acc_path'] = model_file_path
            writer.add_scalars('t_val', {
                f't_val_acc_f{fold_num}': t_val_acc,
                f't_val_recall_f{fold_num}': t_val_recall,
                f't_val_precision_f{fold_num}': t_val_precision,
                f't_val_f1_f{fold_num}': t_val_f1,
            }, step // args.labeled_bs)

            labeled_loader, labeled_set = generate_loader(x_input=x_labeled, y_input=y_labeled,
                                                          batch_size=args.labeled_bs,
                                                          device_no=args.device_no, strategy=None)
            labeled_iter = iter(labeled_loader)
            x_l, y_l = labeled_iter.next()

        try:
            x_u, y_u = unlabeled_iter.next()
        except:
            unlabeled_loader, unlabeled_set = generate_loader(x_input=x_unlabeled, y_input=y_unlabeled,
                                                              device_no=args.device_no,
                                                              batch_size=args.unlabeled_bs, strategy=None)
            unlabeled_iter = iter(unlabeled_loader)
            x_u, y_u = unlabeled_iter.next()

        teacher_optim.zero_grad()
        student_optim.zero_grad()

        # x_l, y_l, x_u, y_u
        bs_l = x_l.shape[0]
        x_t = torch.cat((x_l, x_u))
        t_logits = teacher_net(x_t)
        t_logits_l = t_logits[:bs_l]
        t_logits_u = t_logits[bs_l:]
        hard_pseudo_label = (t_logits_u.detach() > 0.5).float()  # 这里和标注的gt有差别
        t_loss_l = criterion(t_logits_l, y_l)

        # forward and backward pass on student net.
        x_s = torch.cat((x_l, x_u))
        s_logits = student_net(x_s)
        s_logits_l = s_logits[:bs_l]
        s_logits_u = s_logits[bs_l:]

        s_loss_l_old = F.binary_cross_entropy(s_logits_l.detach(), y_l)  # no gradient

        s_loss_u = criterion(s_logits_u, hard_pseudo_label)
        s_loss_u.backward()
        student_optim.step()

        with torch.no_grad():
            s_logits_l = student_net(x_l)
        s_loss_l_new = F.binary_cross_entropy(s_logits_l.detach(), y_l)  # no gradient

        dot_product = s_loss_l_new - s_loss_l_old
        moving_dot_product = moving_dot_product * 0.99 + dot_product * 0.01
        dot_product = dot_product - moving_dot_product

        t_loss_mpl_ori = F.binary_cross_entropy(t_logits_u.detach(), hard_pseudo_label)
        t_loss_mpl = dot_product * t_loss_mpl_ori
        t_loss = t_loss_l + t_loss_mpl
        t_loss.backward()
        teacher_optim.step()

        if args.c_val_prob:
            val_set = DataModule(x=x_valid, y=y_valid)
            val_loader = DataLoader(dataset=val_set, batch_size=args.labeled_bs,
                                    collate_fn=lambda b: collate_func(b, device_no=args.device_no))
            y_arr = predict_res(net=teacher_net, data_loader=val_loader)
            y_val_list.append(y_arr)
        if args.c_train_prob:
            temp_set = DataModule(x=x_labeled, y=y_labeled)
            temp_loader = DataLoader(dataset=temp_set, batch_size=args.labeled_bs,
                                     collate_fn=lambda b: collate_func(b, device_no=args.device_no))
            y_temp = predict_res(net=teacher_net, data_loader=temp_loader)
            y_train_list.append(y_temp)

    import numpy as np
    if args.c_val_prob:
        y_val_list = np.array(y_val_list)
        np.save(f'{args.board_dir}/y_val_arr_steps.arr', y_val_list)
    if args.c_train_prob:
        y_train_list = np.array(y_train_list)
        np.save(f'{args.board_dir}/y_train_arr_steps.arr', y_train_list)

    return score_dict
