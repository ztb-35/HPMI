from .defender import Defender
from .strip_defender import STRIPDefender, BadnetSTRIPDefender
from .rap_defender import RAPDefender, MHBATRAPDefender
from .onion_defender import ONIONDefender
from .fp_defender import FinePruningDefender

DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'strip_badnet': BadnetSTRIPDefender,
    'rap': RAPDefender,
    'mh_rap': MHBATRAPDefender,
    'onion': ONIONDefender,
    'finepruning': FinePruningDefender,
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
