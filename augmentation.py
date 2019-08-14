import cv2
from mrcnn import utils
import skimage
import imageio
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import matplotlib.pyplot as plt
file_path = './test'
image_list = os.listdir(file_path)
#mask_list = [image for image in image_list if image.split('.')[1] == 'png']
image_list = sorted([image for image in image_list if image.split('.')[1]!='png'])

image_list = image_list[1:]

    
for img in image_list:
    # Load image and mask
    print(img)
    mask_path = os.path.join('./test', img.split('.')[0] + '_color_mask.png')
    image = skimage.io.imread(os.path.join(file_path,img))
    h,w,_ = image.shape
    #print('image_id:',image_id)
    #print('image size',image.shape)
    segmap = skimage.io.imread(mask_path).astype('float32')
    segmap = SegmentationMapOnImage(segmap, nb_classes=3, shape=image.shape)
    ia.imshow(segmap.draw_on_image(image))
    
    
    # Define our augmentation pipeline.
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        #iaa.Dropout([0.05, 0.2]),      # drop 5% or 20% of all pixels
        #iaa.Sharpen((0.0, 1.0)),       # sharpen the image
        iaa.Affine(rotate=(-30, 30)),  # rotate by -30 to 30 degrees (affects segmaps)
        #iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
        iaa.PerspectiveTransform(scale=(0.01, 0.1))
    ], random_order=True)
    
    # Augment images and segmaps.
    images_aug = []
    segmaps_aug = []
    for k in range(3):
        images_aug_i, segmaps_aug_i = seq(image=image, segmentation_maps=segmap)
        mask = SegmentationMapOnImage.get_arr_int(segmaps_aug_i)
        mask = mask.astype(np.bool)
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
                cv2.imwrite(os.path.join('./train_aug','aug'+str(k)+'_'+img), images_aug_i)
                cv2.imwrite(os.path.join('./mask_aug','aug'+str(k)+'_'+img.split('.')[0] + '_color_mask.png'), 
                            segmaps_aug_i.draw(size=(h,w)))
                
                images_aug.append(images_aug_i)
                segmaps_aug.append(segmaps_aug_i)
        
        
    
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