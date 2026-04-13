from models.TimeMoE import Model as TimeMoEBackbone
from models._pretrained_solar_physics import FrozenPretrainedSolarPhysicsModel


class Model(FrozenPretrainedSolarPhysicsModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'TimeMoESolarPhysics',
            TimeMoEBackbone(configs),
            configs.enc_in * 32,
        )
