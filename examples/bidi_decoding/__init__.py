# from .bidi_decoder_layer import *
import importlib
import os


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        model_name = file[: file.find(".py")]
        try:
            importlib.import_module(
                "examples.bidi_decoding." + model_name
            )
        except:
            importlib.import_module(
                "bidi_decoding." + model_name
            )
