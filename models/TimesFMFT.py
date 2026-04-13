from models.TimesFM import Model as TimesFMBackbone
from models._pretrained_adapter_ft import FrozenPretrainedAdapterFTModel


class Model(FrozenPretrainedAdapterFTModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'TimesFMFT',
            TimesFMBackbone(configs),
            configs.enc_in * 32,
        )
