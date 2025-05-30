# --------------------------------------------------------------------------------------
# WholeBrain base folder!
#
# By Gustavo Patow
#
# --------------------------------------------------------------------------------------
from sys import platform

if platform == "win32":
    WorkBrainDataFolder = "C:/Users/Usuario/PycharmProjects/TFG/Data_Raw/"
else:
    raise Exception('Unrecognized OS!!!')
