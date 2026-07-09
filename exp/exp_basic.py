import os
import importlib

import torch


class LazyModelModule:
    def __init__(self, module_name):
        self.module_name = module_name
        self.module = None

    def _load(self):
        if self.module is None:
            self.module = importlib.import_module('models.{}'.format(self.module_name))
        return self.module

    def __getattr__(self, name):
        return getattr(self._load(), name)


def lazy_model(module_name):
    return LazyModelModule(module_name)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': lazy_model('TimesNet'),
            'TimesNetSolarPhysics': lazy_model('TimesNetSolarPhysics'),
            'Autoformer': lazy_model('Autoformer'),
            'AutoformerSolarPhysics': lazy_model('AutoformerSolarPhysics'),
            'Chronos': lazy_model('Chronos'),
            'ChronosSolarPhysics': lazy_model('ChronosSolarPhysics'),
            'Chronos2': lazy_model('Chronos2'),
            'Chronos2SolarPhysics': lazy_model('Chronos2SolarPhysics'),
            'Transformer': lazy_model('Transformer'),
            'TransformerSolarPhysics': lazy_model('TransformerSolarPhysics'),
            'Nonstationary_Transformer': lazy_model('Nonstationary_Transformer'),
            'Nonstationary_TransformerSolarPhysics': lazy_model('Nonstationary_TransformerSolarPhysics'),
            'DLinear': lazy_model('DLinear'),
            'DLinearSolarPhysics': lazy_model('DLinearSolarPhysics'),
            'EMAformer': lazy_model('EMAformer'),
            'EMAformerSolarPhysics': lazy_model('EMAformerSolarPhysics'),
            'FEDformer': lazy_model('FEDformer'),
            'FEDformerSolarPhysics': lazy_model('FEDformerSolarPhysics'),
            'Informer': lazy_model('Informer'),
            'InformerSolarPhysics': lazy_model('InformerSolarPhysics'),
            'LightTS': lazy_model('LightTS'),
            'LightTSSolarPhysics': lazy_model('LightTSSolarPhysics'),
            'Reformer': lazy_model('Reformer'),
            'ReformerSolarPhysics': lazy_model('ReformerSolarPhysics'),
            'ETSformer': lazy_model('ETSformer'),
            'ETSformerSolarPhysics': lazy_model('ETSformerSolarPhysics'),
            'PatchTST': lazy_model('PatchTST'),
            'PatchTSTSolarPhysics': lazy_model('PatchTSTSolarPhysics'),
            'Pyraformer': lazy_model('Pyraformer'),
            'PyraformerSolarPhysics': lazy_model('PyraformerSolarPhysics'),
            'MICN': lazy_model('MICN'),
            'MICNSolarPhysics': lazy_model('MICNSolarPhysics'),
            'mixlinear': lazy_model('mixlinear'),
            'mixlinearSolarPhysics': lazy_model('mixlinearSolarPhysics'),
            'Crossformer': lazy_model('Crossformer'),
            'CrossformerSolarPhysics': lazy_model('CrossformerSolarPhysics'),
            'FiLM': lazy_model('FiLM'),
            'FiLMSolarPhysics': lazy_model('FiLMSolarPhysics'),
            'iTransformer': lazy_model('iTransformer'),
            'iTransformerSolarPhysics': lazy_model('iTransformerSolarPhysics'),
            'Koopa': lazy_model('Koopa'),
            'KoopaSolarPhysics': lazy_model('KoopaSolarPhysics'),
            'TiDE': lazy_model('TiDE'),
            'TiDESolarPhysics': lazy_model('TiDESolarPhysics'),
            'FreTS': lazy_model('FreTS'),
            'FreTSSolarPhysics': lazy_model('FreTSSolarPhysics'),
            'MambaSimple': lazy_model('MambaSimple'),
            'MambaSimpleSolarPhysics': lazy_model('MambaSimpleSolarPhysics'),
            'TimeMixer': lazy_model('TimeMixer'),
            'TimeMixerSolarPhysics': lazy_model('TimeMixerSolarPhysics'),
            'TimeMoE': lazy_model('TimeMoE'),
            'TimeMoESolarPhysics': lazy_model('TimeMoESolarPhysics'),
            'TSMixer': lazy_model('TSMixer'),
            'TSMixerSolarPhysics': lazy_model('TSMixerSolarPhysics'),
            'SegRNN': lazy_model('SegRNN'),
            'Sundial': lazy_model('Sundial'),
            'SundialSolarPhysics': lazy_model('SundialSolarPhysics'),
            'TemporalFusionTransformer': lazy_model('TemporalFusionTransformer'),
            "SCINet": lazy_model('SCINet'),
            'SCINetSolarPhysics': lazy_model('SCINetSolarPhysics'),
            'PAttn': lazy_model('PAttn'),
            'PAttnSolarPhysics': lazy_model('PAttnSolarPhysics'),
            'TiRex': lazy_model('TiRex'),
            'TiRexSolarPhysics': lazy_model('TiRexSolarPhysics'),
            'TimeXer': lazy_model('TimeXer'),
            'TimeXerSolarPhysics': lazy_model('TimeXerSolarPhysics'),
            'TimesFM': lazy_model('TimesFM'),
            'TimesFMSolarPhysics': lazy_model('TimesFMSolarPhysics'),
            'WPMixer': lazy_model('WPMixer'),
            'MultiPatchFormer': lazy_model('MultiPatchFormer'),
            'KANAD': lazy_model('KANAD'),
            'MSGNet': lazy_model('MSGNet'),
            'TimeFilter': lazy_model('TimeFilter'),
            'TimeFilterSolarPhysics': lazy_model('TimeFilterSolarPhysics'),
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            self.model_dict['Mamba'] = lazy_model('Mamba')

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if not self.args.use_multi_gpu:
                # Respect any externally configured CUDA visibility and always
                # use the first visible device as the local runtime device.
                device = torch.device('cuda:0')
                if visible_devices:
                    print('Use GPU: cuda:0 (visible devices: {}, requested local gpu: {})'.format(
                        visible_devices, self.args.gpu
                    ))
                else:
                    print('Use GPU: cuda:{} (no CUDA_VISIBLE_DEVICES override)'.format(self.args.gpu))
            else:
                if not visible_devices:
                    os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                    visible_devices = self.args.devices
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{} (visible: {})'.format(self.args.gpu, visible_devices))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
