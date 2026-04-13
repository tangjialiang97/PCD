from models.Sundial import Model as SundialBackbone
from models._pretrained_solar_physics import FrozenPretrainedSolarPhysicsModel


class Model(FrozenPretrainedSolarPhysicsModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'SundialSolarPhysics',
            SundialBackbone(configs),
            configs.enc_in * 32,
        )
