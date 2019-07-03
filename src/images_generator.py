import os
import glob
import pprint
import numpy as np
from PIL import Image, ImageOps, ImageMath, ImageChops, ImageEnhance
import cv2
import matplotlib.pyplot as plt
# リアルタイムにデータ拡張しながら画像データのパッチを生成する
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils


# 画像を読み込む関数(RGB/GRAYとINDEX画像で処理を変える)
def load_images(dir_name, image_shape,seed_num=0,Test=False):
    files = get_file_list_name(dir_name)
    files.sort()
    h, w, ch = image_shape
    images = []
    for i, file in enumerate(files):
#        src_img = Image.open(os.path.join(dir_name,file+'.png')).crop((3,3,w+3,h+3))
        src_img = Image.open(os.path.join(dir_name,file+'.png'))
        img_array = np.asarray(src_img,dtype=np.float32)
        img_array = np.minimum(img_array, 255)
        img_array = np.reshape(img_array, image_shape)
        images.append(img_array / 255)
    return (files, np.array(images, dtype=np.float32))

# 画像を読み込む関数(RGB/GRAYとINDEX画像で処理を変える)
def load_list(dir_name, name_list, image_shape,index_shape,Test=False):
    h, w, ch       = image_shape
    images = []
    masks  = []
    idximg = []
#     crop_x = 3 
#     crop_y = 3 
#     crop_w = crop_x + w 
#     crop_h = crop_y + h 

    for i, file in enumerate(name_list):
                src_img  = Image.open(os.path.join(dir_name, file +'.png'))
                src_msk  = Image.open(os.path.join(dir_name, file +'_seg.png'))

                if not(src_msk.mode == 'P'):
                        print('This mask image is not index mode!!')
                        pause
                if Test==False:
                        # ランダムくロッピング
                        # if np.random.rand() < 0.5:
                        #         crop_x = np.random.randint(0,6) 
                        #         crop_y = np.random.randint(0,6) 
                        #         crop_w = crop_x + w 
                        #         crop_h = crop_y + h 
                        # 黒目のみ色相彩度変更
                        if np.random.rand() < 0.5:
                                eff_img  = src_img.copy()     
                                hue, s, v = src_img.convert("HSV").split()
                                if np.random.rand() < 0.5:
                                        hue = ImageMath.eval("(hue + num) % 255", hue=hue,num=np.random.randint(0,359,1)).convert("L")
                                if np.random.rand() < 0.5:
                                        s = ImageMath.eval("(s + num) % 255", s=s,num=np.random.randint(-15,25,1)).convert("L")
                                eff_img  = Image.merge("HSV", (hue, s, v)).convert("RGB")
                                msk_img  = Image.new('1',src_msk.size)
                                msk_arry = np.array(src_msk, dtype=np.uint8)
                                msk_img  = np.where(((msk_arry == 1) | (msk_arry == 4)),255,0)
                                msk_img  = np.array(msk_img, dtype=np.uint8)
                                msk      = Image.fromarray(msk_img).convert('1',)
                                src_img  = Image.composite(eff_img,src_img, msk)

                        # 明暗をスクリーンと乗算で実施
                        if np.random.rand() < 0.5:
                                eff_img  = src_img.copy()     
                                if np.random.rand() < 0.5:
                                      eff_img  = ImageChops.screen(eff_img, eff_img)            
                                      src_img  = Image.blend(src_img,eff_img, np.random.uniform(0.1, 0.75, 1))
                                else:
                                      eff_img  = ImageChops.multiply(eff_img, eff_img)           
                                      src_img  = Image.blend(src_img,eff_img, np.random.uniform(0.05, 0.25, 1))
                        # コントラスト補正
                        if np.random.rand() < 0.5:
                                enhancer = ImageEnhance.Contrast(src_img)
                                src_img  = enhancer.enhance(np.random.uniform(0.925, 1.0, 1))            
                        # 色相シフト±5
                        if np.random.rand() < 0.5:
                                hue, s, v = src_img.convert("HSV").split()
                                _hue = ImageMath.eval("(hue + num) % 255", hue=hue,num=np.random.randint(-5,5,1)).convert("L")
                                src_img  = Image.merge("HSV", (_hue, s, v)).convert("RGB")

#                src_img = src_img.crop((crop_x,crop_y,crop_w,crop_h))
#                src_msk = src_msk.crop((crop_x,crop_y,crop_w,crop_h))

                img_array  = np.array(src_img, dtype=np.float32)
                msk_array  = np.array(src_msk, dtype=np.uint8)
                # 学習時の水増し(numpy配列　左右反転とランダムノイズ)
                if Test==False:
                        if np.random.rand() < 0.5:
                                noise_img = np.random.normal(0,3,[128,128,3])
                                img_array = img_array+noise_img
                        if np.random.rand() < 0.5:
                                img_array = img_array[:,::-1]
                                msk_array = np.fliplr(msk_array)

                img_array = np.minimum(img_array, 255)

                images.append(img_array / 255.0)
                # masks.append(msk_array)
                index_img = np.zeros(index_shape,dtype='uint8')
                for y in range(h):
                    for x in range(w):
                        index_img[x,y,msk_array[x][y]]=1
                idximg.append(index_img)
    return idximg, images

def load_index(dir_name, index_shape,seed_num=0,Test=False):
    files = glob.glob(os.path.join(dir_name, '*.png'))
    files.sort()
    w, h, class_num = index_shape
    images = []
    fcnt = len(files)
    np.random.seed(seed_num)
    randnum = np.random.rand(fcnt)
    for i, file in enumerate(files):
        src_img = Image.open(file)
        if Test == False:
           if randnum[i] < 0.5:
              src_img = ImageOps.mirror(src_img)
        img_array = np.asarray(src_img)
        if src_img.mode == 'P':
            index_img = np.zeros(index_shape)
            for y in range(h):
                for x in range(w):
                    index_img[x,y,img_array[x][y]]=1
            images.append(index_img)
            
    return files, np.array(images, dtype=np.uint8)

def copy_images(dir_name, dir_load_name, file_name_list):
#    pprint.pprint(file_name_list)        
    for file_name in file_name_list:
        name = os.path.basename(file_name)
        im   = Image.open(os.path.join(dir_load_name, name+'.png'))        
        save_path = os.path.join(dir_name, name+'.png')
        im.save(save_path)
        print('saved : ' , save_path)


def save_images(dir_name, image_data_list, file_name_list):
#    pprint.pprint(file_name_list)        
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        name = os.path.basename(file_name)
        save_path = os.path.join(dir_name, name+'.png')
        img = np.minimum(image_data[:,:,::-1]*255,255)        
        cv2.imwrite(save_path,img)
        print('saved : ' , save_path)

def save_index(dir_name, image_data_list, file_name_list):
#    pprint.pprint(file_name_list)        
    for _, (image_data, file_name) in enumerate(zip(image_data_list, file_name_list)):
        im = Image.open('..\\template\\mask.png')
#        category_list = {'others':[0,0,0],
#                        'lip_upper':[128,0,0],
#                        'lip_lower':[128,128,0],
#                        'lip_inner':[0,128,0]
#                        }
        name = os.path.basename(file_name) +'_seg.png'
        #print('image.mode',np.round(image_data,1))
        (w, h, class_num) = image_data.shape
        image_data = np.reshape(image_data, (w, h, class_num))
        index_img  = np.asarray(np.zeros([w,h,1]))
        output_img = Image.new("P",[128,128],0)
        # output_img = Image.new("P",[134,134],0)
#        print(index_img.shape)
        for y in range(h):
            for x in range(w):
                index_img[x,y,0] = np.argmax(image_data[x,y],axis=0)
        distImg = np.reshape(np.uint8(index_img),[w,h])
        outImg = Image.fromarray(np.uint8(distImg),mode="P")
        output_img.paste(outImg,(0,0))
        # output_img.paste(outImg,(3,3))
        outImg = output_img
        outImg.putpalette(im.getpalette())
#        outImg.putpalette(category_list)
        save_path = os.path.join(dir_name, name)
        outImg.save(save_path)
        print('saved : ' , save_path)

# リストから指定拡張子のみを抜いた名前を取得
def get_file_list_name(dir_name):
        flag = 0
        files = glob.glob(os.path.join(dir_name, '*_seg.png'))
        if not files:
                files = glob.glob(os.path.join(dir_name, '*.png'))
                flag = 1
        file_name = []
        check_name = []
        for i in files:
                if flag == 1:
                        file_list = i.split('.png')[0]
                else:                
                        file_list = i.split('_seg.png')[0]
                name  = os.path.basename(file_list)
                # 重複名を削除
                if len(check_name)==0:
                        check_name = name
                        file_name.append(name)
                else:
                        if not (name == check_name):
                                file_name.append(name)
        return file_name

def get_batch(dir_name, batch_size,input_image_shape,input_index_shape,Test=False,Debug=False):
    
        # 指定ディレクトリ内の指定拡張子のみのリストを取得
        file_name = get_file_list_name(dir_name)
#        pprint.pprint(file_name)
        np.random.seed(0)
        np.random.shuffle(file_name)

        while True:

                # 画像をリスト順に読み込む
                images = []
                idxmsk = []

                eof_num = len(file_name)
                # batchごとに読み込む
                for idx in range(0,eof_num,batch_size):
                        if idx > eof_num - batch_size:
                                rem_num = eof_num - idx
                                s_num = np.random.randint(0,eof_num - rem_num+1)
                                batch_fname = file_name[idx:idx+eof_num]+file_name[s_num:s_num+rem_num-1]
                        else:
                                batch_fname = file_name[idx:idx+batch_size]
                        # for i in range(0,batch_size):
                        idxmsk,images = load_list(dir_name, batch_fname, input_image_shape,input_index_shape,Test=False)
                        # save_images('Outputs', images, batch_fname)
                        # save_index('Outputs', idxmsk, batch_fname)
                        images = np.array(images,dtype=np.float32)
                        idxmsk = np.array(idxmsk,dtype=np.uint8)
 
                        yield images,idxmsk
 

