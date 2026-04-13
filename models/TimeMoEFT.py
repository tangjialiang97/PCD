from models.TimeMoE import Model as TimeMoEBackbone
from models._pretrained_adapter_ft import FrozenPretrainedAdapterFTModel


class Model(FrozenPretrainedAdapterFTModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'TimeMoEFT',
            TimeMoEBackbone(configs),
            configs.enc_in * 32,
        )
