from models.Chronos import Model as ChronosBackbone
from models._pretrained_solar_physics import FrozenPretrainedSolarPhysicsModel


class Model(FrozenPretrainedSolarPhysicsModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'ChronosSolarPhysics',
            ChronosBackbone(configs),
            configs.enc_in * 32,
        )
