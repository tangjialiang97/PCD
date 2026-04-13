from models.Sundial import Model as SundialBackbone
from models._pretrained_adapter_ft import FrozenPretrainedAdapterFTModel


class Model(FrozenPretrainedAdapterFTModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'SundialFT',
            SundialBackbone(configs),
            configs.enc_in * 32,
        )
