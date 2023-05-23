import logging
import logging.config
import yaml
import os

path_to_config = os.path.abspath(os.path.dirname(__file__))
logging.config.dictConfig(yaml.load(open(os.path.join(path_to_config, 'logger_config.yaml'), 'r'), Loader=yaml.FullLoader))
SleepInferenceLogger = logging.getLogger('SleepInferenceLogger')