# Created by pengsheng.huang at 8/14/2020
from enum import Enum, unique


@unique
class TrainMode(Enum):
    ALL = 0
    GAN = 1
    AE = 2
    AAE = 3
    VAEGAN = 4
    FL = 5


I = 0
J = 0

LR = 1e-4
LAMBDA = 0.9
TRAIN_DATA_PATH = '/data2/huangps/data_freakie/gx/data/gonet_original/data_train'
TEST_DATA_PATH = '/data2/huangps/data_freakie/gx/data/gonet_original/data_test'

FL_TRAIN_DATA_PATH = '/data2/huangps/data_freakie/gx/data/gonet_original/data_train_annotation'
FL_TEST_DATA_PATH = '/data2/huangps/data_freakie/gx/data/gonet_original/data_test_annotation'

MODEL_FILE_GEN = 'nn_model/featlayer_gen_single.h5'
MODEL_FILE_DIS = 'nn_model/featlayer_dis_single.h5'
MODEL_FILE_INVG = 'nn_model/featlayer_invg_single.h5'
MODEL_FILE_FL = 'nn_model/classlayer_core.h5'

NZ = 100

# centER OF PICTUre
XC = 310
YC = 321

YOFFSET = 310
XOFFSET = 310
XYOFFSET = 275
XYC = [(XC - XYOFFSET, YC - XYOFFSET),
       (XC + XYOFFSET, YC + XYOFFSET)]

# resiZE PARAMETErs
RSIZEX = 128
RSIZEY = 128
EPOCHS = 50
BATCH_SIZE = 32
