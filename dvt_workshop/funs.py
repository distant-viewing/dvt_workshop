def dvt_detectron_config():
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog, DatasetCatalog
    from detectron2.engine import DefaultPredictor
    from detectron2 import model_zoo

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    predictor = DefaultPredictor(cfg)
    md = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    return predictor, md

def dvt_tidy_instance_data(predictions, md, ofile):
    import pandas as pd

    instances = predictions["instances"]
    class_names = md.thing_classes

    boxes = instances.get("pred_boxes").tensor.cpu().numpy()
    cls = [class_names[x] for x in instances.get("pred_classes").cpu().numpy()]
    scores = instances.get("scores").cpu().numpy()

    df = pd.DataFrame({
        "path": ofile,
        "class": cls,
        "prob": scores,
        "x0": boxes[:, 0],
        "y0": boxes[:, 1],
        "x1": boxes[:, 2],
        "y1": boxes[:, 3]
    })

    return df


def dvt_show_instance_predictions(predictions, md, im):
    from detectron2.utils.visualizer import Visualizer
    from google.colab.patches import cv2_imshow

    instances = predictions["instances"]
    v = Visualizer(
        im[:, :, ::-1], md, scale=1.2
    )
    out = v.draw_instance_predictions(instances.to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


def dvt_load_embed_image_model(layer="fc2"):
    from tensorflow.keras.applications.vgg19 import VGG19
    from tensorflow.keras.models import Model

    base_model = VGG19(weights='imagenet')
    embed = Model(
        inputs=base_model.input, outputs=base_model.get_layer(layer).output
    )

    return embed


def dvt_embed_image(path, embed):
    import numpy as np
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.vgg19 import preprocess_input

    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_embed = embed.predict(x)

    return img_embed
