from torch.utils.data import DataLoader
import torch.utils.data as torch_data
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from loss.lion import Lion
from models import Informer, Autoformer, Transformer, MyModel, Linear, NLinear, DLinear, MyModel1
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')
def normalized(data, normalize_method, norm_statistic=None):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
        data = (data - norm_statistic['min']) / scale
        data = np.clip(data, 0.0, 1.0)
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = (data - mean) / std
        norm_statistic['std'] = std
    return data, norm_statistic


def de_normalized(data, normalize_method, norm_statistic):
    if normalize_method == 'min_max':
        if not norm_statistic:
            norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
        scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
        data = data * scale + norm_statistic['min']
    elif normalize_method == 'z_score':
        if not norm_statistic:
            norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
        mean = norm_statistic['mean']
        std = norm_statistic['std']
        std = [1 if i == 0 else i for i in std]
        data = data * std + mean
    return data

class ForecastDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        target_data = self.data[hi:hi + self.horizon] #12:24
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
        return x_end_idx

class ForecastTestDataset(torch_data.Dataset):
    def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
        self.window_size = window_size # 12
        self.interval = interval  #1
        self.horizon = horizon
        self.normalize_method = normalize_method
        self.norm_statistic = norm_statistic
        df = pd.DataFrame(df)
        df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
        self.data = df
        self.df_length = len(df)
        self.x_end_idx = self.get_x_end_idx()
        if normalize_method:
            self.data, _ = normalized(self.data, normalize_method, norm_statistic)

    def __getitem__(self, index):
        hi = self.x_end_idx[index] #12
        lo = hi - self.window_size #0
        train_data = self.data[lo: hi] #0:12
        target_data = self.data[hi:hi + self.horizon] #12:24
        x = torch.from_numpy(train_data).type(torch.float)
        y = torch.from_numpy(target_data).type(torch.float)
        return x, y

    def __len__(self):
        return len(self.x_end_idx)

    def get_x_end_idx(self):
        # each element `hi` in `x_index_set` is an upper bound for get training data
        # training data range: [lo, hi), lo = hi - window_size
        x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
        x_end_idx = [x_index_set[j * 12] for j in range((len(x_index_set)) // 12)]
        return x_end_idx

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'MyModel': MyModel,
            'MyModel1': MyModel1,
            'NLinear': NLinear,
            'Linear': Linear,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self):
        data_file = os.path.join('./dataset/PEMS', self.args.data_path + '.npz')
        print('data file:',data_file)
        data = np.load(data_file,allow_pickle=True)
        data = data['data'][:,:,0]
        train_ratio = self.args.train_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        valid_ratio = self.args.valid_length / (self.args.train_length + self.args.valid_length + self.args.test_length)
        test_ratio = 1 - train_ratio - valid_ratio
        train_data = data[:int(train_ratio * len(data))]
        valid_data = data[int(train_ratio * len(data)):int((train_ratio + valid_ratio) * len(data))]
        test_data = data[int((train_ratio + valid_ratio) * len(data)):]
        if len(train_data) == 0:
            raise Exception('Cannot organize enough training data')
        if len(valid_data) == 0:
            raise Exception('Cannot organize enough validation data')
        if len(test_data) == 0:
            raise Exception('Cannot organize enough test data')
        if self.args.normtype == 0: # we follow StemGNN and other related works for somewhat fair comparison (orz..), but we strongly suggest use self.args.normtype==2!!!
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_mean = np.mean(valid_data, axis=0)
            val_std = np.std(valid_data, axis=0)
            val_normalize_statistic = {"mean": val_mean.tolist(), "std": val_std.tolist()}
            test_mean = np.mean(test_data, axis=0)
            test_std = np.std(test_data, axis=0)
            test_normalize_statistic = {"mean": test_mean.tolist(), "std": test_std.tolist()}
        elif self.args.normtype == 1:
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            train_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            val_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
            test_normalize_statistic = {"mean": data_mean.tolist(), "std": data_std.tolist()}
        else:
            train_mean = np.mean(train_data, axis=0)
            train_std = np.std(train_data, axis=0)
            train_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            val_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
            test_normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
        train_set = ForecastDataset(train_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                normalize_method=self.args.norm_method, norm_statistic=train_normalize_statistic)
        valid_set = ForecastDataset(valid_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                    normalize_method=self.args.norm_method, norm_statistic=val_normalize_statistic)
        test_set = ForecastTestDataset(test_data, window_size=self.args.window_size, horizon=self.args.horizon,
                                    normalize_method=self.args.norm_method, norm_statistic=test_normalize_statistic)
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size, drop_last=False, shuffle=True,
                                            num_workers=1)
        valid_loader = DataLoader(valid_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False, num_workers=1)
        node_cnt = train_data.shape[1]
        return train_set, train_loader, test_set, test_loader, valid_set, valid_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # model_optim = Lion(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'MyModel' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'MyModel' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_set, train_loader, test_set, test_loader, valid_set, valid_loader = self._get_data()
        # if not self.args.train_only:
        #     vali_data, vali_loader = self._get_data(flag='val')
        #     test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)
                # batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'MyModel' in self.args.model:
                            outputs = self.model(batch_x)
                        # else:
                        #     if self.args.output_attention:
                        #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        #     else:
                        #         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'MyModel' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(valid_set, valid_loader, criterion)
                test_loss = self.vali(test_set, test_loader, criterion)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        train_set, train_loader, test_set, test_loader, valid_set, valid_loader = self._get_data()
        
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'MyModel' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'MyModel' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
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
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = np.concatenate(preds, axis=0)
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        
        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # np.save(folder_path + 'real_prediction.npy', preds)
        pd.DataFrame(np.append(np.transpose([pred_data.future_dates]), preds[0], axis=1), columns=pred_data.cols).to_csv(folder_path + 'real_prediction.csv', index=False)

        return
