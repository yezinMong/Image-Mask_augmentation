import cv2
from mrcnn import utils
import skimage
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import matplotlib.pyplot as plt
import os

file_path = './train'
image_list = os.listdir(file_path)
#mask_list = [image for image in image_list if image.split('.')[1] == 'png']
image_list = sorted([image for image in image_list if image.split('.')[1]!='png'])

image_list = image_list[1:]
#image_list = random.sample(image_list, k=10)
    
for img in image_list:
    # Load image and mask
    mask_path = os.path.join('./mask', img.split('.')[0] + '_color_mask.png')
    image = skimage.io.imread(os.path.join(file_path,img))
    h,w,_ = image.shape
    image = np.pad(image, ((100,100), (100,100), (0, 0)), mode='constant')
    segmap = skimage.io.imread(mask_path).astype('float32')
    segmap = np.pad(segmap, ((100,100), (100,100), (0, 0)), mode='constant')
    segmap = SegmentationMapOnImage(segmap, nb_classes=3, shape=image.shape)
    
    
    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        #iaa.PadToFixedSize(height=h+100,width=w+100, position='center'),
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.AddToHueAndSaturation((-50, 50)), # change hue and saturation
        #iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        #iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-90, 90)),  # rotate by -30 to 30 degrees (affects segmaps)
        #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        iaa.PerspectiveTransform(scale=(0.04, 0.1)),
        iaa.PiecewiseAffine(scale=(0.04, 0.05))
    ], random_order=False)
    
    # Augment images and segmaps.
    images_aug = []
    segmaps_aug = []
    for k in range(5):
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        #print('image_aug_i size',images_aug_i.shape)
        mask = SegmentationMapOnImage.get_arr_int(segmaps_aug_i)
        #print('segmaps_aug_i size',mask.shape)
        mask = mask.astype(np.bool)
        h,w = mask.shape
        #plt.imshow(mask)
        #plt.show()
        mask = mask.reshape([h, w, 1])
        
        # mask가 이미지 크기 밖으로 넘어가지 않을 경우에만 출력
        if mask[0].any() or mask[-1].any():
            pass
        else:
            mask = np.transpose(mask, axes=(1,0,2))
            if mask[0].any() or mask[-1].any():
                pass
            else:
                mask = np.transpose(mask, axes=(1,0,2))
                cv2.imwrite(os.path.join('./train_aug','aug2_'+str(k)+'_'+img), images_aug_i)
                cv2.imwrite(os.path.join('./mask_aug','aug2_'+str(k)+'_'+img.split('.')[0] + '_color_mask.png'), 
                            segmaps_aug_i.draw(size=(h,w)))
                
                #images_aug.append(images_aug_i)
                #segmaps_aug.append(segmaps_aug_i)
        
        
    
    # We want to generate an image of original input images and segmaps
    # before/after augmentation.
    # It is supposed to have five columns: (1) original image, (2) original
    # image with segmap, (3) augmented image, (4) augmented
    # segmap on augmented image, (5) augmented segmap on its own in.
    # We now generate the cells of these columns.
    
    
    if images_aug:
        cells = []
        for image_aug, segmap_aug in zip(images_aug, segmaps_aug):
            cells.append(image)                                      # column 1
            cells.append(segmap.draw_on_image(image))                # column 2
            cells.append(image_aug)                                  # column 3
            cells.append(segmap_aug.draw_on_image(image_aug))        # column 4
            cells.append(segmap_aug.draw(size=image_aug.shape[:2]))  # column 5

        # Convert cells to grid image and save.
        grid_image = ia.draw_grid(cells, cols=5)
        ia.imshow(grid_image)
        #imageio.imwrite(os.path.join('./test','aug'+img), grid_image)
    
