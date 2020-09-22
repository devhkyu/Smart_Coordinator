from source.train.MaskingClothes import mask_clothes as mc
from glob import glob

maskingModel = mc.Model(threshold=0.7)

keyword = ['date', 'formal', 'school', 'street']
for key in keyword:
    inx = 0
    for x in glob('../../data/image/evaluation/'+key+'/*'):
        print("Directory: {}".format(x))
        img, masked_img, label_type, label, score, complete, combine = maskingModel.run(x)
        if complete is True:
            print("Label: {}".format(label))
            print("Complete: {}".format(complete))
            # combine.save('../../data/image/evaluation/'+key+'_masked/'+str(inx)+'.jpg')
            inx += 1