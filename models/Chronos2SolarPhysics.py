from models.Chronos2 import Model as Chronos2Backbone
from models._pretrained_solar_physics import FrozenPretrainedSolarPhysicsModel


class Model(FrozenPretrainedSolarPhysicsModel):
    def __init__(self, configs):
        super().__init__(
            configs,
            'Chronos2SolarPhysics',
            Chronos2Backbone(configs),
            configs.enc_in * 32,
        )
