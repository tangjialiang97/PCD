from models.Chronos import Model as ChronosBackbone
from models._pretrained_adapter_ft import FrozenPretrainedAdapterFTModel


class Model(FrozenPretrainedAdapterFTModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'ChronosFT',
            ChronosBackbone(configs),
            configs.enc_in * 32,
        )
