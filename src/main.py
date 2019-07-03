import os
import glob
import numpy as np
import pprint
from datetime import datetime
from keras import optimizers
from fcn_net import FcnNet

from images_generator import get_batch,get_file_list_name,save_images,save_index,load_list,load_index,load_images,copy_images
from option_parser import get_option
from keras.callbacks import TensorBoard, LearningRateScheduler,ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from keras_to_tf import savetf
############################################################################
CLASS_NUM = 4
INPUT_IMAGE_SHAPE = (128,128,3)
INPUT_INDEX_SHAPE = (128,128,CLASS_NUM)
BATCH_SIZE = 35
EPOCHS = 0
START_EPOCHS = 0 
END_EPOCHS   = START_EPOCHS + EPOCHS
SAVE_EPOCHSTEP = 100
####################################################################
# 現在時刻の取得
now = datetime.now()

####################################################################
# ディレクトリの設定
DIR_MODEL    = os.path.join('..', 'Model')
DIR_MOCKP    = os.path.join(DIR_MODEL, 'ckp')

DIR_DATA   = os.path.join('..', 'Data')

DIR_TRAIN    = os.path.join(DIR_DATA, 'Inputs')
DIR_OUTPUTS  = os.path.join(DIR_DATA, 'Outputs')

DIR_VALID    = os.path.join(DIR_DATA, 'Valid')
DIR_TESTS    = os.path.join(DIR_DATA, 'Test')
DIR_TESTS_ANS= os.path.join(DIR_DATA, 'Test_Answer')

####################################################################
# 保存ファイル/読み込みファイルの指定
File_WEIGHT = 'eye_mask_weight_{0:%Y%m%d}.h5'.format(now)
File_MODEL  = 'eye_mask_model_{0:%Y%m%d}.h5'.format(now)

Load_WEIGHT = 'eye_mask_weight_20190605.h5'
Load_MODEL  = 'eye_mask_model_20190605.h5'

####################################################################
# コールバックの設定［LOGフォルダの生成］
callbacks = []
DIR_LOGS = os.path.join('./logs','{0:%Y%m%d}'.format(now))
if not(os.path.exists(DIR_LOGS)):
    os.mkdir(DIR_LOGS)
####################################################################
# Tensorboardの設定
tb_cb=TensorBoard(log_dir=DIR_LOGS)
callbacks.append(tb_cb)

####################################################################
# アーリーストッピングの設定
# earlystopper = EarlyStopping(monitor='loss', patience=350, verbose=1,min_delta=1.0e-11)
# callbacks.append(earlystopper)

####################################################################
# 学習率のスケジュール
# def lr_decay(EPOCHS):
#     if   EPOCHS <  50: lr = 0.01550
#     elif EPOCHS < 100: lr = 0.00550
#     elif EPOCHS < 150: lr = 0.00150
#     elif EPOCHS < 250: lr = 0.00075
#     elif EPOCHS < 350: lr = 0.00050
#     elif EPOCHS < 500: lr = 0.00025
#     elif EPOCHS < 700: lr = 0.00010
#     else             : lr = 0.00005
#     return lr
# lr_cb = LearningRateScheduler(lr_decay,verbose=0)
# callbacks.append(lr_cb)
####################################################################
# チェックポイント
cp_cb = ModelCheckpoint(filepath=os.path.join(DIR_MOCKP, 'eye_mask_weight_.{epoch:05d}.h5'),save_weights_only=True, period = SAVE_EPOCHSTEP)
callbacks.append(cp_cb)
####################################################################
# 学習率の減衰
cp_cb = ReduceLROnPlateau(monitor='loss', factor=0.75, patience=250, verbose=1, mode='auto', epsilon=1.0e-20, cooldown=0, min_lr=0)
callbacks.append(cp_cb)

####################################################################
# 訓練
def train(INPUT_IMAGE_SHAPE,CLASS_NUM,DIR_MODEL):
    K.clear_session()
    # 学習オプションの設定/モデル構築
    opt     = optimizers.Adam(lr=0.0100,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.00065,amsgrad=False)
    network = FcnNet(INPUT_IMAGE_SHAPE, CLASS_NUM)
    model = network.model()
    model.load_weights(os.path.join(DIR_MODEL, Load_WEIGHT))

    ####################################################################
    # 構築したモデルをターミナルに出力
    model.summary()

    # モデルのコンパイル
    model.compile(optimizer=opt, 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # 学習
    history = model.fit_generator(
                        generator=get_batch(DIR_TRAIN,BATCH_SIZE,INPUT_IMAGE_SHAPE,INPUT_INDEX_SHAPE), 
                        steps_per_epoch=int(len(glob.glob(os.path.join(DIR_TRAIN, '*_seg.png')))/BATCH_SIZE), 
                        initial_epoch = START_EPOCHS,
                        epochs = END_EPOCHS,
                        verbose=2,
                        # workers = 4,
                        # use_multiprocessing = True,
                        validation_data=get_batch(DIR_VALID,BATCH_SIZE,INPUT_IMAGE_SHAPE,INPUT_INDEX_SHAPE),
                        validation_steps=int(len(glob.glob(os.path.join(DIR_VALID, '*_seg.png')))/BATCH_SIZE),
                        callbacks = callbacks)

    #モデルの重み保存/モデルの保存
    model.save_weights(os.path.join(DIR_MODEL, File_WEIGHT))
    model.save(os.path.join(DIR_MODEL, File_MODEL))
####################################################################
# 予測
def predict(input_dir,output_dir=DIR_OUTPUTS):
    K.clear_session()
    (file_names, inputs) = load_images(input_dir, INPUT_IMAGE_SHAPE,Test=True)
    network = FcnNet(INPUT_IMAGE_SHAPE,CLASS_NUM)
    model = network.model()
    model.load_weights(os.path.join(DIR_MODEL, File_WEIGHT))
    preds = model.predict(inputs, BATCH_SIZE)
    #print(preds.shape)
    copy_images(output_dir, input_dir, file_names)
#    save_images(output_dir, inputs, file_names)
    save_index(output_dir, preds, file_names)

############################################################################
if __name__ =='__main__':
    
    if not(os.path.exists(DIR_MODEL)):
        os.mkdir(DIR_MODEL)
    if not(os.path.exists(DIR_MOCKP)):
        os.mkdir(DIR_MOCKP)
    if not(os.path.exists(DIR_OUTPUTS)):
        os.mkdir(DIR_OUTPUTS)
    
    train(INPUT_IMAGE_SHAPE,CLASS_NUM,DIR_MODEL)
    predict(DIR_TRAIN)
    predict(DIR_TESTS,DIR_TESTS_ANS)
    savetf(DIR_MODEL, File_MODEL)