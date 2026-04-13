from models.Chronos2 import Model as Chronos2Backbone
from models._pretrained_adapter_ft import FrozenPretrainedAdapterFTModel


class Model(FrozenPretrainedAdapterFTModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'Chronos2FT',
            Chronos2Backbone(configs),
            configs.enc_in * 32,
        )
