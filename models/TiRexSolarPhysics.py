from models.TiRex import Model as TiRexBackbone
from models._pretrained_solar_physics import FrozenPretrainedSolarPhysicsModel


class Model(FrozenPretrainedSolarPhysicsModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'TiRexSolarPhysics',
            TiRexBackbone(configs),
            configs.enc_in * 32,
        )
