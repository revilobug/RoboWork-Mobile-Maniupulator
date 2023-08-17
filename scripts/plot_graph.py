from time import time
from tensorflow.keras.callbacks import TensorBoard

import datetime
log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
