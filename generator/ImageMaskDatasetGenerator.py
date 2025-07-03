# Copyright 2023-2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2023/08/10 
# 2025/07/03 Modified to generatee PNG files from the original nii files.
# create_base_dataset.py

import os
import sys
import shutil
import cv2

import glob
import numpy as np
import math
import nibabel as nib
import traceback

# Read file
"""
scan = nib.load('/path/to/stackOfimages.nii.gz')
# Get raw data
scan = scan.get_fdata()
print(scan.shape)
(num, width, height)
"""

# See : https://github.com/neheller/kits19/blob/master/starter_code/visualize.py
# BGR color-order
KIDNEY_COLOR = [255, 0, 0]  #Blue
TUMOR_COLOR  = [0, 0, 255]  #Red


# The functions in this class were taken from visualize.py
# https://github.com/neheller/kits19/blob/master/starter_code/visualize.py
#
class ImageMaskDatasetGenerator:

  def __init__(self, width=512, height=512):
    self.width       = width
    self.height      = height
    self.file_format = ".png"

  def class_to_color(self, segmentation, kidney_color, tumor_color, tumor_only=False):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], 3), dtype=np.float32)

    # set output to appropriate color at each location
    # 2023/08/10 antillia.com
    if tumor_only:
      # Set a kidney mask color to be black 
      kidney_color = [0, 0, 0]
    seg_color[np.equal(segmentation, 1)] = kidney_color
    seg_color[np.equal(segmentation, 2)] = tumor_color
    return seg_color

  def create_mask_files(self, niigz, output_dir, index):
    print("--- niigz {}".format(niigz))
    nii = nib.load(niigz)
    print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    #data = np.asanyarray(nii.dataobj)
    num_images = data.shape[0] # math.floor(data.shape[2]/2)
    print("--- num_images {}".format(num_images))
    num = 0
    for i in range(num_images):
      img = data[i, :, :]
      img = self.class_to_color(img, KIDNEY_COLOR, TUMOR_COLOR) 
      img = np.array(img)
      if np.any(img > 0):
        filepath = os.path.join(output_dir, str(index) + "_" + str(i) + self.file_format)
        img = cv2.resize(img, (self.width, self.height))
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        num += 1
    return num
  
  def create_image_files(self, niigz, output_masks_dir, output_images_dir, index):
    print("--- create_image_files nii {}".format(niigz))
    nii = nib.load(niigz)
    print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    #data = np.asanyarray(nii.dataobj)
    num_images = data.shape[0] # math.floor(data.shape[2]/2)
    print("--- num_images {}".format(num_images))
    num = 0
    for i in range(num_images):
      img = data[i, :, :]
   
      filename = str(index) + "_" + str(i) + self.file_format
      mask_filepath = os.path.join(output_masks_dir, filename)
      if os.path.exists(mask_filepath):
        filepath = os.path.join(output_images_dir, filename)
        img = cv2.resize(img, (self.width, self.height))
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        num += 1
    return num
  

  def generate(self, data_dir, output_images_dir, output_masks_dir):
    dirs = glob.glob(data_dir)
    print("--- num dirs {}".format(len(dirs)))
    index = 10000
    for dir in dirs:
      print("== dir {}".format(dir))
      image_nii_file = os.path.join(dir, "imaging.nii")
      seg_nii_file   = os.path.join(dir, "segmentation.nii")
      index += 1
      if os.path.exists(image_nii_file) and os.path.exists(seg_nii_file):
        num_segmentations = self.create_mask_files(seg_nii_file,   output_masks_dir,  index)
        num_images        = self.create_image_files(image_nii_file, output_masks_dir, output_images_dir, index)
        print(" image_nii_file: {}  seg_nii_file: {}".format(num_images, num_segmentations))

        if num_images != num_segmentations:
          raise Exception("The number of images and segmentations is different.")
      else:
        print("Not found segmentation file {} corresponding to {}".format(seg_nii_file, image_nii_file))


if __name__ == "__main__":
  try:
    data_dir          = "./data/case_*"
    output_images_dir = "./Kits19-PNG-master/images/"
    output_masks_dir  = "./Kits19-PNG-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    os.makedirs(output_masks_dir)

    # Create png image and mask files from nii files under data_dir.
    generator = ImageMaskDatasetGenerator()
    generator.generate(data_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


