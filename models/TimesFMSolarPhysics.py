from models.TimesFM import Model as TimesFMBackbone
from models._pretrained_solar_physics import FrozenPretrainedSolarPhysicsModel


class Model(FrozenPretrainedSolarPhysicsModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'TimesFMSolarPhysics',
            TimesFMBackbone(configs),
            configs.enc_in * 32,
        )
