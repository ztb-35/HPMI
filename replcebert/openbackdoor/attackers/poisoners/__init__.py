from .poisoner import Poisoner
from .badnets_poisoner import BadNetsPoisoner


POISONERS = {
    "base": Poisoner,
    "badnets": BadNetsPoisoner,
}

def load_poisoner(config):
    return POISONERS[config["name"].lower()](**config)
