# iMaterialist Fashion 2019 at FGVC6
![Kaggle3](https://user-images.githubusercontent.com/44195740/70390630-f9713900-1a0f-11ea-8f26-1212a2f18536.jpg)
- https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data
- train.csv
- train.zip
- test.zip
- label_descriptions.json (In git, we already have)

# Mask_RCNN
- https://github.com/matterport/Mask_RCNN

# Weight file
- fashion20190930T0958 (1.42GB)
- http://naver.me/x8484YRo
- mask_rcnn_coco.h5 (245.6MB)
- http://naver.me/FeJBJRKJ

# Module Version
- tensorflow (1.14.0) - If version of tensorflow is the latest, some errors will occur.
- keras (2.3.0) - If you want to run server(main.py), you should install keras (2.2.5)
<code>
        pip install keras==2.2.5
</code>

# Errors
- AttributeError: 'Model' object has no attribute 'metrics_tensors'
- Mask_RCNN/mrcnn/model.py line 2191:

        # Add metrics for losses
        for name in loss_names:
            self.keras_model.metrics_tensors = []   # You should add this code
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                tf.reduce_mean(layer.output, keepdims=True)
                * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
