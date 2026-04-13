import os
import torch
from models import (
    Autoformer,
    AutoformerSolarPhysics,
    Chronos,
    Chronos2,
    Chronos2FT,
    Chronos2SolarPhysics,
    ChronosFT,
    ChronosSolarPhysics,
    Crossformer,
    CrossformerSolarPhysics,
    DLinear,
    DLinearReservoirPhysics,
    DLinearSolarPhysics,
    DLinearSolarPhysics3,
    DLinearWindPhysics,
    DLinearWindPhysicsV2,
    DLinearWindPhysicsV3,
    DLinearWindPhysicsV4,
    DLinearWindPhysicsV5,
    DLinearWindPhysicsV6,
    ETSformer,
    ETSformerSolarPhysics,
    FEDformer,
    FEDformerSolarPhysics,
    FiLM,
    FiLMSolarPhysics,
    FreTS,
    FreTSSolarPhysics,
    Informer,
    InformerSolarPhysics,
    KANAD,
    Koopa,
    KoopaSolarPhysics,
    LightTS,
    LightTSSolarPhysics,
    MICN,
    MICNSolarPhysics,
    MSGNet,
    MambaSimple,
    MambaSimpleSolarPhysics,
    MultiPatchFormer,
    Nonstationary_Transformer,
    Nonstationary_TransformerSolarPhysics,
    PAttn,
    PAttnSolarPhysics,
    PatchTST,
    PatchTSTSolarPhysics,
    PatchTSTSolarPhysics3,
    Pyraformer,
    PyraformerSolarPhysics,
    Reformer,
    ReformerSolarPhysics,
    SCINet,
    SCINetSolarPhysics,
    SegRNN,
    Sundial,
    SundialFT,
    SundialSolarPhysics,
    TSMixer,
    TSMixerSolarPhysics,
    TemporalFusionTransformer,
    TiRex,
    TiRexFT,
    TiRexSolarPhysics,
    TiDE,
    TiDESolarPhysics,
    TimeFilter,
    TimeFilterSolarPhysics,
    TimeMixer,
    TimeMixerSolarPhysics,
    TimeMoE,
    TimeMoEFT,
    TimeMoESolarPhysics,
    TimsFMFT,
    TimeXer,
    TimeXerSolarPhysics,
    TimesFM,
    TimesFMFT,
    TimesFMSolarPhysics,
    TimesNet,
    TimesNetSolarPhysics,
    Transformer,
    TransformerSolarPhysics,
    WPMixer,
    iTransformer,
    iTransformerSolarPhysics,
    iTransformerWindPhysics,
    iTransformerWindPhysicsV2,
    iTransformerWindPhysicsV3,
)


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'TimesNetSolarPhysics': TimesNetSolarPhysics,
            'Autoformer': Autoformer,
            'AutoformerSolarPhysics': AutoformerSolarPhysics,
            'Chronos': Chronos,
            'ChronosFT': ChronosFT,
            'ChronosSolarPhysics': ChronosSolarPhysics,
            'Chronos2': Chronos2,
            'Chronos2FT': Chronos2FT,
            'Chronos2SolarPhysics': Chronos2SolarPhysics,
            'Transformer': Transformer,
            'TransformerSolarPhysics': TransformerSolarPhysics,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'Nonstationary_TransformerSolarPhysics': Nonstationary_TransformerSolarPhysics,
            'DLinear': DLinear,
            'DLinearReservoirPhysics': DLinearReservoirPhysics,
            'DLinearSolarPhysics': DLinearSolarPhysics,
            'DLinearSolarPhysics3': DLinearSolarPhysics3,
            'DLinearWindPhysics': DLinearWindPhysics,
            'DLinearWindPhysicsV2': DLinearWindPhysicsV2,
            'DLinearWindPhysicsV3': DLinearWindPhysicsV3,
            'DLinearWindPhysicsV4': DLinearWindPhysicsV4,
            'DLinearWindPhysicsV5': DLinearWindPhysicsV5,
            'DLinearWindPhysicsV6': DLinearWindPhysicsV6,
            'FEDformer': FEDformer,
            'FEDformerSolarPhysics': FEDformerSolarPhysics,
            'Informer': Informer,
            'InformerSolarPhysics': InformerSolarPhysics,
            'LightTS': LightTS,
            'LightTSSolarPhysics': LightTSSolarPhysics,
            'Reformer': Reformer,
            'ReformerSolarPhysics': ReformerSolarPhysics,
            'ETSformer': ETSformer,
            'ETSformerSolarPhysics': ETSformerSolarPhysics,
            'PatchTST': PatchTST,
            'PatchTSTSolarPhysics': PatchTSTSolarPhysics,
            'PatchTSTSolarPhysics3': PatchTSTSolarPhysics3,
            'Pyraformer': Pyraformer,
            'PyraformerSolarPhysics': PyraformerSolarPhysics,
            'MICN': MICN,
            'MICNSolarPhysics': MICNSolarPhysics,
            'Crossformer': Crossformer,
            'CrossformerSolarPhysics': CrossformerSolarPhysics,
            'FiLM': FiLM,
            'FiLMSolarPhysics': FiLMSolarPhysics,
            'iTransformer': iTransformer,
            'iTransformerSolarPhysics': iTransformerSolarPhysics,
            'iTransformerWindPhysics': iTransformerWindPhysics,
            'iTransformerWindPhysicsV2': iTransformerWindPhysicsV2,
            'iTransformerWindPhysicsV3': iTransformerWindPhysicsV3,
            'Koopa': Koopa,
            'KoopaSolarPhysics': KoopaSolarPhysics,
            'TiDE': TiDE,
            'TiDESolarPhysics': TiDESolarPhysics,
            'FreTS': FreTS,
            'FreTSSolarPhysics': FreTSSolarPhysics,
            'MambaSimple': MambaSimple,
            'MambaSimpleSolarPhysics': MambaSimpleSolarPhysics,
            'TimeMixer': TimeMixer,
            'TimeMixerSolarPhysics': TimeMixerSolarPhysics,
            'TimeMoE': TimeMoE,
            'TimeMoEFT': TimeMoEFT,
            'TimeMoESolarPhysics': TimeMoESolarPhysics,
            'TSMixer': TSMixer,
            'TSMixerSolarPhysics': TSMixerSolarPhysics,
            'SegRNN': SegRNN,
            'Sundial': Sundial,
            'SundialFT': SundialFT,
            'SundialSolarPhysics': SundialSolarPhysics,
            'TemporalFusionTransformer': TemporalFusionTransformer,
            "SCINet": SCINet,
            'SCINetSolarPhysics': SCINetSolarPhysics,
            'PAttn': PAttn,
            'PAttnSolarPhysics': PAttnSolarPhysics,
            'TiRex': TiRex,
            'TiRexFT': TiRexFT,
            'TiRexSolarPhysics': TiRexSolarPhysics,
            'TimeXer': TimeXer,
            'TimeXerSolarPhysics': TimeXerSolarPhysics,
            'TimesFM': TimesFM,
            'TimesFMFT': TimesFMFT,
            'TimsFMFT': TimsFMFT,
            'TimesFMSolarPhysics': TimesFMSolarPhysics,
            'WPMixer': WPMixer,
            'MultiPatchFormer': MultiPatchFormer,
            'KANAD': KANAD,
            'MSGNet': MSGNet,
            'TimeFilter': TimeFilter,
            'TimeFilterSolarPhysics': TimeFilterSolarPhysics,
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

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
