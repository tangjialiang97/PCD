from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.utils
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
import itertools
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _uses_solar_internal_physics(self):
        return bool(getattr(self.args, 'use_internal_physics', False)) or self.args.model.endswith('SolarPhysics')

    def _uses_solar_persistence(self):
        return self.args.model in {
            'SimplePersistence',
            'RCRFPM',
            'RCRF-PM',
            'SmartPersistence',
            'RPM',
            'R-PM',
            'CAPM',
            'CA-PM',
            'CFPM',
            'CF-PM',
        }

    def _extract_future_zenith(self, batch_y):
        return batch_y[:, -self.args.pred_len:, -1]

    def _select_main_channels(self, batch_tensor):
        return batch_tensor[:, :, :self.args.enc_in]

    def _select_supervised_channels(self, batch_tensor):
        f_dim = -1 if self.args.features == 'MS' else 0
        return batch_tensor[:, :, f_dim:]

    def _forward_model(self, batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, data_set):
        if self._uses_solar_internal_physics():
            future_zenith = self._extract_future_zenith(batch_y).to(batch_x_main.device)
            return self.model(
                batch_x_main,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                future_zenith=future_zenith,
            )
        if self._uses_solar_persistence():
            future_zenith = self._extract_future_zenith(batch_y).to(batch_x_main.device)
            return self.model(
                batch_x_main,
                batch_x_mark,
                dec_inp,
                batch_y_mark,
                future_zenith=future_zenith,
            )
        return self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        optimizer_param_models = {
            'ChronosSolarPhysics',
            'Chronos2SolarPhysics',
            'SundialSolarPhysics',
            'TiRexSolarPhysics',
            'TimesFMSolarPhysics',
            'TimeMoESolarPhysics',
        }
        if self.args.model in optimizer_param_models:
            parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        else:
            parameters = self.model.parameters()
        model_optim = optim.Adam(parameters, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        if self.args.debug:
            vali_loader = list(itertools.islice(vali_loader, 10))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, vali_data
                        )
                else:
                    outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, vali_data)
                outputs = self._select_supervised_channels(outputs[:, -self.args.pred_len:, :])
                batch_y_main_pred = self._select_supervised_channels(
                    batch_y_main[:, -self.args.pred_len:, :].to(self.device)
                )

                pred = outputs.detach()
                true = batch_y_main_pred.detach()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.debug:
            train_loader = list(itertools.islice(train_loader, 10))
            vali_loader = list(itertools.islice(vali_loader, 10))
            test_loader = list(itertools.islice(test_loader, 10))

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, train_data)
                outputs = self._select_supervised_channels(outputs[:, -self.args.pred_len:, :])

                batch_y_main_pred = self._select_supervised_channels(
                    batch_y_main[:, -self.args.pred_len:, :].to(self.device)
                )
                loss = criterion(outputs, batch_y_main_pred)

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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            checkpoint_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

        preds = []
        trues = []


        self.model.eval()
        if self.args.debug:
            test_loader = list(itertools.islice(test_loader, 10))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self._forward_model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark, batch_y, test_data)
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_main_pred = batch_y_main[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y_main_pred = batch_y_main_pred.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y_main_pred.shape
                    if outputs.shape[-1] != batch_y_main_pred.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y_main_pred.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_main_pred = test_data.inverse_transform(batch_y_main_pred.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = self._select_supervised_channels(outputs)
                batch_y_main_pred = self._select_supervised_channels(batch_y_main_pred)

                pred = outputs
                true = batch_y_main_pred

                preds.append(pred)
                trues.append(true)
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        is_three = bool(getattr(self.args, 'is_three', False))
        if not is_three:
            preds = preds[:, :, 0:1]
            trues = trues[:, :, 0:1]
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            dtw_preds = preds[:, :, 0:1]
            dtw_trues = trues[:, :, 0:1]
            for i in range(dtw_preds.shape[0]):
                x = dtw_preds[i].reshape(-1, 1)
                y = dtw_trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        ymax = test_data.ymax
  
        f = open("result.txt", 'a')
        f.write(setting + "  \n")

     
        ghi_ymax = ymax[0] if isinstance(ymax, (list, tuple, np.ndarray)) else ymax
        mae, mse, nrmse, nmae = metric(preds, trues, ghi_ymax)
        print('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))
        f.write('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))

        f.write('\n')
        f.write('\n')
        f.close()
        return
