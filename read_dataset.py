# -*- coding: utf-8 -*-
import struct
import numpy as np
import os
import six.moves.cPickle as pickle
from alcon_utils import AlconUtils
import cv2


class PRMUDataSet:
    def __init__(self, data_type):
        self.data_dir_path = "../alcon2017prmu/dataset/"
        self.data = None
        self.target = None
        self.dump_name = data_type
        self.image_size = 96
        self.categories = -1
        self.n_types_target = -1
        self.hash_target = None

    def load_data_target(self):
        #if os.path.exists(self.dump_name + '1'):
        #    self.load_dataset()
        if self.target is None:
            self.target = []
            self.data = []
            self.hash_target = []
            # 初期化
            alcon = AlconUtils(self.data_dir_path)

            # アノテーションの読み込み
            fn = "target_lv" + self.dump_name + ".csv"
            alcon.load_annotations_target(fn)

            fn = "groundtruth_lv" + self.dump_name + ".csv"
            alcon.load_annotations_ground(fn)

            for bb_id, target in alcon.targets.items():
                img_filename = alcon.get_filename_char(bb_id)
                #print(img_filename)
                code = alcon.ground_truth[bb_id]
                # Load an color image in grayscale
                img = cv2.imread(img_filename, 0)
                #print(shape(img))
                height, width = img.shape
                WHITE = [255, 255, 255]
                if height > width:
                    img = cv2.copyMakeBorder(img, 0, 0, (height - width) // 2, int(np.ceil((height - width) / 2)),
                                             cv2.BORDER_CONSTANT,
                                             value=WHITE)
                else:
                    img = cv2.copyMakeBorder(img, (width - height) // 2, int(np.ceil((width - height) / 2)), 0, 0,
                                             cv2.BORDER_CONSTANT,
                                             value=WHITE)
                
                #print(img.shape)
                img.resize(self.image_size, self.image_size,1)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.data.append(img)

                # determine index of target of this data
                index_target = -1
                try:
                    index_target = self.hash_target.index(code)
                except ValueError:
                    self.hash_target.append(code)
                    index_target = self.hash_target.index(code)
                self.target.append(index_target)

            self.data = np.array(self.data, np.float32)
            self.target = np.array(self.target, np.int32)
            self.categories = len(self.hash_target)
            print (self.data.shape)
            print (self.target.shape)
            print (self.categories)
            #self.dump_dataset()


    def get_n_types_target(self):
        if self.target is None:
            self.load_data_target()

        if self.n_types_target is not -1:
            return self.n_types_target

        return len(self.target)


    def dump_dataset(self):
        data1, data2, data3, data4, data5, data6 = np.array_split(self.data, 6)
        target1, target2, target3, target4, target5, target6 = np.array_split(self.target, 6)

        pickle.dump((data1, target1), open(self.dump_name + '1', 'wb'), -1)
        pickle.dump((data2, target2), open(self.dump_name + '2', 'wb'), -1)
        pickle.dump((data3, target3), open(self.dump_name + '3', 'wb'), -1)
        pickle.dump((data4, target4), open(self.dump_name + '4', 'wb'), -1)
        pickle.dump((data5, target5), open(self.dump_name + '5', 'wb'), -1)
        pickle.dump((data6, target6), open(self.dump_name + '6', 'wb'), -1)


    def load_dataset(self):
        data1, target1 = pickle.load(open(self.dump_name + '1', 'rb'))
        data2, target2 = pickle.load(open(self.dump_name + '2', 'rb'))
        data3, target3 = pickle.load(open(self.dump_name + '3', 'rb'))
        data4, target4 = pickle.load(open(self.dump_name + '4', 'rb'))
        data5, target5 = pickle.load(open(self.dump_name + '5', 'rb'))
        data6, target6 = pickle.load(open(self.dump_name + '6', 'rb'))

        self.data = np.concatenate((data1, data2, data3, data4, data5, data6), axis=0)
        self.target = np.concatenate((target1, target2, target3, target4, target5, target6), axis=0)