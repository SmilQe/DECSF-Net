from thop import profile, clever_format

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import iMMTST
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler

import os
import time

import warnings
import numpy as np
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')


class Exp_Main_MM(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main_MM, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'iMMTST': iMMTST,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (water_seq_x, water_mark_x, mete_seq_x, mete_mark_x, water_seq_y, water_mark_y) in enumerate(
                    vali_loader):

                water_seq_x = water_seq_x.float().to(self.device)
                water_mark_x = water_mark_x.float().to(self.device)

                mete_seq_x = mete_seq_x.float().to(self.device)
                mete_mark_x = mete_mark_x.float().to(self.device)

                water_seq_y = water_seq_y.float()
                water_mark_y = water_mark_y.float().to(self.device)

                # decoder input
                water_dec_inp = torch.zeros_like(water_seq_y[:, -self.args.pred_len:, :]).float()
                water_dec_inp = torch.cat([water_seq_y[:, :self.args.label_len, :], water_dec_inp], dim=1).float().to(
                    self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                        else:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                else:
                    if self.args.output_attention:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                    else:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                water_seq_y = water_seq_y[:, -self.args.pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = water_seq_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        writer = SummaryWriter("./tf_logs")

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='vali')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:  # args.use_amp=False 是否启用混合精度
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (water_seq_x, water_mark_x, mete_seq_x, mete_mark_x, water_seq_y, water_mark_y) in enumerate(
                    train_loader):
                iter_count += 1
                model_optim.zero_grad()

                water_seq_x = water_seq_x.float().to(self.device)
                water_mark_x = water_mark_x.float().to(self.device)
                mete_seq_x = mete_seq_x.float().to(self.device)
                mete_mark_x = mete_mark_x.float().to(self.device)
                water_seq_y = water_seq_y.float().to(self.device)
                water_mark_y = water_mark_y.float().to(self.device)

                water_dec_inp = torch.zeros_like(
                    water_seq_y[:, -self.args.pred_len:, :]).float()  # [B x pred_len x features]
                water_dec_inp = torch.cat([water_seq_y[:, :self.args.label_len, :], water_dec_inp], dim=1).float().to(
                    self.device)  # [B x label_len x features]

                if epoch == i == 0:
                    print('########################  Paramters and FLOPs  ######################### ')

                    # total_params (trainable parameters + freezing trainable parameters
                    total_params = sum(p.numel() for p in self.model.parameters())
                    total_params += sum(p.numel() for p in self.model.buffers())
                    print(f"{total_params:,} total parameters")
                    print(f"{total_params / (1024 * 1024):.2f}M total parameters")
                    print(f"{total_params / (1024 * 1024):.2f}M total parameters")
                    total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print(f"{total_trainable_params:,} training parameters")
                    print(f"{total_params / (1024 * 1024):.2f}M training parameters")


                    flops, _ = profile(self.model, inputs=(
                        water_seq_x[[0]], water_mark_x[[0]], mete_seq_x[[0]], mete_mark_x[[0]]))
                    macs = clever_format(flops, "%.3f")
                    print(f"{macs:}  FLOPs")
                    print('####################################################################### ')

                # mm_transformer
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                        else:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                        outputs = outputs[:, -self.args.pred_len:, :]  # model predict value
                        water_seq_y = water_seq_y[:, -self.args.pred_len:, :].to(self.device)
                        loss = criterion(outputs, water_seq_y)
                        train_loss.append(loss.item())

                else:
                    if self.args.output_attention:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                    else:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                    outputs = outputs[:, -self.args.pred_len:, :]  # model predict value
                    water_seq_y = water_seq_y[:, -self.args.pred_len:, :].to(self.device)
                    loss = criterion(outputs, water_seq_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            writer.add_scalar("Loss/train", scalar_value=train_loss, global_step=epoch + 1)
            writer.add_scalar("Loss/vali", scalar_value=vali_loss, global_step=epoch + 1)
            writer.add_scalar("Loss/test", scalar_value=test_loss, global_step=epoch + 1)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        writer.close()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, 'checkpoint.pth')), strict=False)

        preds = []
        trues = []
        inputx = []
        folder_path = "./test_results/bs1"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        min_loss = 1e9
        min_i = 0
        with torch.no_grad():
            for i, (water_seq_x, water_mark_x, mete_seq_x, mete_mark_x, water_seq_y, water_mark_y) in enumerate(
                    test_loader):

                water_seq_x = water_seq_x.float().to(self.device)
                water_mark_x = water_mark_x.float().to(self.device)

                mete_seq_x = mete_seq_x.float().to(self.device)
                mete_mark_x = mete_mark_x.float().to(self.device)

                water_seq_y = water_seq_y.float().to(self.device)
                water_mark_y = water_mark_y.float().to(self.device)

                # decoder input
                water_dec_inp = torch.zeros_like(water_seq_y[:, -self.args.pred_len:, :])
                water_dec_inp = torch.cat([water_seq_y[:, :self.args.label_len, :], water_dec_inp], dim=1).float().to(
                    self.device)

                # self.model.forward (water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)
                if i == 0:
                    print('########################  Paramters and FLOPs  ######################### ')

                    # total_params (trainable parameters + freezing trainable parameters
                    total_params = sum(p.numel() for p in self.model.parameters())
                    total_params += sum(p.numel() for p in self.model.buffers())
                    print(f"{total_params:,} total parameters")
                    print(f"{total_params / (1024 * 1024):.2f}M total parameters")
                    print(f"{total_params / (1024 * 1024):.2f}M total parameters")
                    total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                    print(f"{total_trainable_params:,} training parameters")
                    print(f"{total_params / (1024 * 1024):.2f}M training parameters")

                    flops, _ = profile(self.model,
                                       inputs=(water_seq_x[[0]], water_mark_x[[0]], mete_seq_x[[0]], mete_mark_x[[0]]))
                    macs = clever_format(flops, "%.3f")
                    print(f"{macs:}  FLOPs")
                    print('####################################################################### ')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                        else:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                else:
                    if self.args.output_attention:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                    else:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                outputs = outputs[:, -self.args.pred_len:, :]
                water_seq_y = water_seq_y[:, -self.args.pred_len:, :].to(self.device)

                mse = nn.MSELoss()
                loss = mse(outputs, water_seq_y)

                if loss < min_loss:
                    min_loss = loss
                    min_i = i

                outputs = outputs.detach().cpu().numpy()
                water_seq_y = water_seq_y.detach().cpu().numpy()

                pred = outputs
                true = water_seq_y

                preds.append(pred)
                trues.append(true)
                inputx.append(water_seq_x.detach().cpu().numpy())

                # if i % 1 == 0:
                #     input = water_seq_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, 0], true[0, :, 0]), axis=0)
                #     pd = np.concatenate((input[0, :, 0], pred[0, :, 0]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, f'{str(i)}_mse_{loss.item()}.pdf'))

            print(f"Min loss: {min_loss}, Min i: {min_i}")

        if self.args.test_flop:
            test_params_flop(self.model, (water_seq_x.shape[1], water_seq_x.shape[2]))
            exit()

        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = "./results/" + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, rse:{}'.format(mse, mae, rmse, mape, rse))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, rse:{}'.format(mse, mae, rmse, mape, rse))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'inputx.npy', inputx)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (water_seq_x, water_mark_x, mete_seq_x, mete_mark_x, water_seq_y, water_mark_y) in enumerate(
                    pred_loader):
                water_seq_x = water_seq_x.float().to(self.device)
                water_mark_x = water_mark_x.float().to(self.device)

                mete_seq_x = mete_seq_x.float().to(self.device)
                mete_mark_x = mete_mark_x.float().to(self.device)

                water_seq_y = water_seq_y.float()
                water_mark_y = water_mark_y.float().to(self.device)

                # decoder input
                water_dec_inp = torch.zeros(
                    [water_seq_y.shape[0], self.args.pred_len, water_seq_y.shape[2]]).float().to(
                    self.device)  # [b, self.pred_len, d_model]
                water_dec_inp = torch.cat([water_seq_y[:, :self.args.label_len, :], water_dec_inp], dim=1).float().to(
                    self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                        else:
                            outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)

                else:
                    if self.args.output_attention:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]
                    else:
                        outputs = self.model(water_seq_x, water_mark_x, mete_seq_x, mete_mark_x)[0]

                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
