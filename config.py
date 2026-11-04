from enum import StrEnum

class Model(StrEnum):
    DISTILBERT = "distilbert"
    DEIT = "deit"
    PYTHIA = "pythia"

class Dataset(StrEnum):
    SST2 = "sst2"
    IMAGENET = "imagenet"

class Approximation(StrEnum):
    NONE = "none"
    ACTIVATION = "activation"
    ATTENTION = "attention"
    BOTH = "both"