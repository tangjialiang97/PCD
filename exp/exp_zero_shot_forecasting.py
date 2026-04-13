from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import itertools
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Zero_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _uses_timesfm_fast_path(self):
        return self.args.model == 'TimesFM'

    def _uses_time_moe_fast_path(self):
        return self.args.model == 'TimeMoE'

    def _uses_tirex_fast_path(self):
        return self.args.model == 'TiRex'

    def _select_main_channels(self, batch_tensor):
        return batch_tensor[:, :, :self.args.enc_in]

    def _select_supervised_channels(self, batch_tensor):
        f_dim = -1 if self.args.features == 'MS' else 0
        return batch_tensor[:, :, f_dim:]

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.args.debug:
            test_loader = list(itertools.islice(test_loader, 10))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # TimesFM converts inputs back to CPU numpy internally before its
                # own batched GPU decode, so keeping the outer tensors on CPU
                # avoids an extra GPU -> CPU -> GPU round trip.
                use_timesfm_fast_path = self._uses_timesfm_fast_path()
                use_time_moe_fast_path = self._uses_time_moe_fast_path()
                use_tirex_fast_path = self._uses_tirex_fast_path()
                if use_timesfm_fast_path:
                    batch_x = batch_x.float()
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float()
                    batch_y_mark = batch_y_mark.float()
                elif use_time_moe_fast_path or use_tirex_fast_path:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()
                    batch_x_mark = batch_x_mark.float()
                    batch_y_mark = batch_y_mark.float()
                else:
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x_main = self._select_main_channels(batch_x)
                batch_y_main = self._select_main_channels(batch_y)

                # decoder icudat
                dec_inp = torch.zeros_like(batch_y_main[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_main[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if not use_timesfm_fast_path and not use_time_moe_fast_path and not use_tirex_fast_path:
                    dec_inp = dec_inp.to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x_main, batch_x_mark, dec_inp, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_main = batch_y_main[:, -self.args.pred_len:, :]
                if not use_timesfm_fast_path and not use_time_moe_fast_path and not use_tirex_fast_path:
                    batch_y_main = batch_y_main.to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y_main = batch_y_main.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y_main.shape
                    if outputs.shape[-1] != batch_y_main.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y_main.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y_main = test_data.inverse_transform(batch_y_main.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = self._select_supervised_channels(outputs)
                batch_y_main = self._select_supervised_channels(batch_y_main)

                pred = outputs
                true = batch_y_main

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    icudat = batch_x_main.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = icudat.shape
                        icudat = test_data.inverse_transform(icudat.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((icudat[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((icudat[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

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
        mae, mse, nrmse, nmae = metric(preds, trues, ymax)
        print('mse:{}, mae:{}, nrmse:{}, nmae:{}, dtw:{}'.format(mse, mae, nrmse, nmae, dtw))
        f = open("result_zero_shot_forecast_" + self.args.data + "_" + self.args.data_path.split(".")[0] + ".txt", 'a')

        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, nrmse:{}, nmae:{}, dtw:{}'.format(mse, mae, nrmse, nmae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        return
