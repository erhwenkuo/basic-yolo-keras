import os
import cv2
import copy
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence
import xml.etree.ElementTree as ET
from utils import BoundBox, normalize, bbox_iou

def parse_annotation(ann_dir, img_dir, labels=[]):
    """解析PASCAL VOC格式的圖像標註檔

    根據PASCAL VOC標註檔存放的目錄路徑迭代地解析每一個標註檔，
    將每個圖像的檔名(filename)、圖像的寬(width)、高(height)、圖像的類別(name)以
    及物體的邊界框的坐標(xmin,ymin,xmax,ymax)擷取出來。

    參數:
        ann_dir: PASCAL VOC標註檔存放的目錄路徑
        img_dir: 圖像檔存放的目錄路徑
        labels: 圖像資料集的物體類別列表

    回傳:
        all_imgs: 一個列表物件, 每一個物件都包括了要訓練用的重要資訊。例如:
                    {
                        'filename': '/tmp/img/img001.jpg',
                        'width': 128,
                        'height': 128,
                        'object':[
                            {'name':'person',xmin:0, ymin:0, xmax:28, ymax:28},
                            {'name':'person',xmin:45, ymin:45, xmax:60, ymax:60}
                        ]
                    }
        seen_labels: 一個字典物件(k:圖像類別, v:出現的次數)用來檢視每一個圖像類別出現的次數
    """
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}
        
        tree = ET.parse(os.path.join(ann_dir, ann))
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = os.path.join(img_dir, elem.text) # 圖像檔的路徑
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'depth' in elem.tag:
                img['depth'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        # 檢看是是否有物體的標誌是沒有在傳入的物體類別(labels)中
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
                        
    return all_imgs, seen_labels

class BatchGenerator(Sequence):
    """用來產生批次訓練資料的類別

    這個類別繼承了keras.utils.Sequence的類別，它主要用在提供模型訓練時的批次資料。
    每個繼承Sequence的類別都必須實現__getitem__和__len__方法。在每一個訓練循環中
    如果想修改數據集，可以通過實現`on_epoch_end`來達成。

    參數:
        images: 圖像Meta物件的列表, 每個meta物件如下:
            {
                'depth': 3,
                'filename': 'D:\\pythonworks\\basic-yolo-keras\\data\\simpson\\images\\abraham_grampa_simpson_pic_0000.jpg',
                'height': 416,
                'object': [
                            {'name': 'abraham_grampa_simpson',
                            'xmax': 52,
                            'xmin': 57,
                            'ymax': 72,
                            'ymin': 72}
                            ],
                'width': 576
            }

        config: 設定物件, 把含很多產生器所需要的參數, 例如:
            {
                'IMAGE_H'         : IMAGE_H, # YOLOv2網絡輸入的image_h
                'IMAGE_W'         : IMAGE_W, # YOLOv2網絡輸入的image_w
                'GRID_H'          : GRID_H,  # 直向網格的拆分數量
                'GRID_W'          : GRID_W,  # 橫向網格的拆分數量
                'BOX'             : BOX,     # 每個單一網格要預測的邊界框數量
                'LABELS'          : LABELS,  # 要預測的圖像種類列表
                'CLASS'           : len(LABELS), # 要預測的圖像種類數
                'ANCHORS'         : ANCHORS, # 每個單一網格要預測的邊界框時用的錨點
                'BATCH_SIZE'      : BATCH_SIZE, # 訓練時的批量數
                'TRUE_BOX_BUFFER' : 50, # 一個訓練圖像最大數量的邊界框數
            }

        norm: 圖像像素歸一化函式
    """
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.counter = 0
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1]) for i in range(int(len(config['ANCHORS'])//2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                #iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                #sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                    #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                    #rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    # 可以產生多少數量的批量數據
    def __len__(self):
        return int(np.ceil(float(len(self.images))/self.config['BATCH_SIZE']))   

    # 取得某特定索引的批量數據
    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        # x_batch(圖像)張量結構 -> [samples, img_height, img_width, img_channels]
        # b_batch(邊界框)張量結構 -> [samples, 1, 1, 1, true_box_buffer, 4]
        # y_batch(網絡輸出) 張量結構 -> [samples, GRID_H, GRID_W, BOX, 4+1+CLASS]
        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],  self.config['GRID_W'], self.config['BOX'], 4+1+self.config['CLASS'])) # desired network output

        # 迭代每一個圖像
        for train_instance in self.images[l_bound:r_bound]:

            # augment input image and fix object's position and size
            # 進行圖像增益並校對圖像裡物體的位置和大小
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            
            # construct output from object's x, y, w, h
            # 構建圖像裡每一個物體的x, y, w, h的輸出參數
            true_box_index = 0
            
            # 迭代圖像裡每一個物體
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and obj['name'] in self.config['LABELS']:
                    # 計算"邊界框"的中心點
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    # 計算"邊界框"的中心點位於那一個Grid的Cell中
                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name']) # 取得物體類別的編碼索引
                        
                        center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_w, center_h]

                        # find the anchor that best predicts this box
                        # 找到最能預測這個盒子的錨點
                        best_anchor = -1
                        max_iou     = -1
                        
                        shifted_box = BoundBox(0, 
                                               0, 
                                               center_w, 
                                               center_h)
                        
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)
                            
                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        # 將前述計算出來的結果x，y，w，h，信心度和類別機率置放到y_batch的結構中
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box # x，y，w，h
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1. # 信心度
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1 # 類別機率
                        
                        # assign the true box to b_batch
                        # 將真正的"邊界框"放置到b_batch的結構中
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # assign input image to x_batch
            # 將輸入圖像的資料置放至x_batch的結構中
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img) # 進行歸一化處理
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img

            # increase instance counter in current batch
            instance_count += 1  

        self.counter += 1
        #print ' new batch created', self.counter

        # 回傳三種訓練所需的資料
        #   - x_batch: 
        #   - b_batch:
        #   - y_batch: 圖像類別的標籤
        return [x_batch, b_batch], y_batch

    # 當每次的訓練循環完成時被呼叫
    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images) # 從新打散序列的順序
        self.counter = 0

    # 進行圖像增強
    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name) # 讀圖像資料
        h, w, c = image.shape
        
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            ### 縮放圖像
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            ### 圖像水平/垂直移動
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            ### 翻轉圖像
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
                
            image = self.aug_pipe.augment_image(image)            
            
        # resize the image to standard size
        # 將圖像調整為標準尺寸
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1] # 轉換 BGR -> RBG

        # fix object's position and size
        # 修正物體在圖像中的位置和大小
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                    
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
                
        return image, all_objs
