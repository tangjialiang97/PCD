from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
<<<<<<< HEAD
from utils.metrics import metric
=======
from utils.metrics import metric, metric_with_ymax
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
import itertools
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

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
        if self.args.debug:
            vali_loader = list(itertools.islice(vali_loader, 10))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

<<<<<<< HEAD
                batch_x_main = batch_x[:, :, :3]
                batch_y_main = batch_y[:, :, :3]

                # decoder input
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.npu.amp.autocast():
                        outputs = self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_main_pred = batch_y_main[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y_main_pred.detach()
=======
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach()
                true = batch_y.detach()
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1

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
<<<<<<< HEAD
            scaler = torch.npu.amp.GradScaler()
=======
            scaler = torch.cuda.amp.GradScaler()
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1

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

<<<<<<< HEAD
                batch_x_main = batch_x[:, :, :3]
                batch_y_main = batch_y[:, :, :3]

                # decoder input
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                outputs = self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]

                batch_y_main_pred = batch_y_main[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y_main_pred)

                if self.args.use_auxiliary:
                    # Physical closure:
                    # GHI_hat ~= DNI_hat * cos(theta_z) + DHI_hat
                    # theta_z is appended as the last feature in batch_y.
                    batch_y_aux = batch_y[:, -self.args.pred_len:, :].to(self.device)
                    zenith_deg = batch_y_aux[:, :, -1]
                    cos_zenith = torch.cos(torch.deg2rad(zenith_deg)).clamp(min=0.0)
                    ghi_hat = outputs[:, :, 0]
                    dni_hat = outputs[:, :, 1]
                    dhi_hat = outputs[:, :, 2]
                    residual = ghi_hat - (dni_hat * cos_zenith + dhi_hat)
                    # Apply closure only in daytime with usable solar elevation,
                    # then use a relative robust penalty so closure acts as a weak regularizer.
                    closure_cos_threshold = float(getattr(self.args, 'closure_day_cos_threshold', 0.2))
                    closure_ref_floor = float(getattr(self.args, 'closure_ref_floor', 100.0))
                    closure_beta = float(getattr(self.args, 'closure_beta', 0.03))
                    closure_warmup_epochs = int(getattr(self.args, 'closure_warmup_epochs', 2))
                    day_mask = (cos_zenith > closure_cos_threshold).float()
                    ghi_true = batch_y_main_pred[:, :, 0].detach()
                    closure_ref = ghi_true.abs().clamp_min(closure_ref_floor)
                    residual_rel = residual / closure_ref
                    closure_term = torch.nn.functional.smooth_l1_loss(
                        residual_rel,
                        torch.zeros_like(residual_rel),
                        reduction='none',
                        beta=closure_beta,
                    )
                    day_mask_sum = day_mask.sum()
                    l_closure = (closure_term * day_mask).sum() / day_mask_sum.clamp_min(1.0)
                    closure_weight = float(getattr(self.args, 'closure_weight', 1.0))
                    if closure_warmup_epochs > 0:
                        closure_weight *= min(1.0, float(epoch + 1) / float(closure_warmup_epochs))
                    loss = loss + closure_weight * l_closure

                    # Meteorological attenuation limit:
                    # GHI_upper = I0 * cos(theta_z) * f(P, RH, T)
                    # L_limit = mean(ReLU(GHI_hat - GHI_upper)^2)
                    i0 = float(getattr(self.args, 'solar_constant', 1367.0))
                    # Column order in Irr data:
                    # [ghi, dni, dhi, air_temp, relhum, press, windsp, winddir, max_windsp, zenith]
                    air_temp = batch_y_aux[:, :, 3]
                    relhum = batch_y_aux[:, :, 4]
                    press = batch_y_aux[:, :, 5]

                    # Build a smooth attenuation factor in [0, 1].
                    relhum_norm = (relhum / 100.0).clamp(0.0, 1.0)
                    press_ratio = (press / 1013.25).clamp(0.6, 1.2)
                    temp_penalty = ((air_temp - 25.0).abs() / 40.0).clamp(0.0, 1.0)
                    f_attn = (press_ratio * (1.0 - 0.3 * relhum_norm) * (1.0 - 0.2 * temp_penalty)).clamp(0.0, 1.0)

                    ghi_upper = i0 * cos_zenith * f_attn
                    l_limit = torch.mean(torch.relu(ghi_hat - ghi_upper) ** 2)
                    limit_weight = float(getattr(self.args, 'limit_weight', 1.0))
                    loss = loss + limit_weight * l_limit

                    # Variability loss:
                    # L_var = exp(-gamma * windsp) * ||dGHI/dt||^2
                    # Low wind speed -> stronger smoothness; high wind speed -> weaker penalty.
                    if outputs.shape[1] > 1:
                        sample_interval = float(getattr(self.args, 'sample_interval', 1.0))
                        delta_t = max(sample_interval, 1.0)
                        dghi_dt = (ghi_hat[:, 1:] - ghi_hat[:, :-1]) / delta_t
                        windsp = batch_y_aux[:, 1:, 6].clamp(min=0.0)
                        gamma_var = float(getattr(self.args, 'var_gamma', 0.1))
                        var_weight_term = torch.exp(-gamma_var * windsp)
                        l_var = torch.mean(var_weight_term * (dghi_dt ** 2))
                        var_weight = float(getattr(self.args, 'var_weight', 1.0))
                        loss = loss + var_weight * l_var

                train_loss.append(loss.item())
=======
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1

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
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

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

<<<<<<< HEAD
                batch_x_main = batch_x[:, :, :3]
                batch_y_main = batch_y[:, :, :3]

                # decoder input
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                outputs = self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
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
      

                outputs = outputs[:, :, f_dim:]
                batch_y_main_pred = batch_y_main_pred[:, :, f_dim:]

                pred = outputs
                true = batch_y_main_pred
=======
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
      

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1

                preds.append(pred)
                trues.append(true)
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
<<<<<<< HEAD

        preds = preds[:, :, 0:1]
        trues = trues[:, :, 0:1]
=======
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        ymax = test_data.ymax
        # ymax = 1.0
<<<<<<< HEAD
        mae, mse, nrmse, nmae = metric(preds, trues, ymax)
        print('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))
        f = open("result_long_term_forecast_" + self.args.data + "_" + self.args.data_path.split(".")[0] + "_" + str(self.args.use_auxiliary) + ".txt", 'a')
=======
        mae, mse, nrmse, nmae = metric_with_ymax(preds, trues, ymax)
        print('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))
        f = open("result_long_term_forecast_" + self.args.data + "_" + self.args.data_path.split(".")[0] + ".txt", 'a')
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, nrmse:{}, nmae:{}'.format(mse, mae, nrmse, nmae))
        f.write('\n')
        f.write('\n')
        f.close()
<<<<<<< HEAD
        return
=======
        return
>>>>>>> 43bc95b0a4ddd08689931ee59d6b5b814a8c9ed1
