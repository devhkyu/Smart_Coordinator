from PIL import Image
import source.train.MaskingClothes.mask_clothes as mask

model = mask.Model(img_size=512, threshold=0.7, gpu_count=1, images_per_gpu=1)
img, masked_image, label_type, label, score, complete = model.run('test1.jpg')

"""
# Print Image
for x in masked_image:
    x.show()
"""