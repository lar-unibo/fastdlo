import logging
import numpy as np
import torch
import cv2 

import fastdlo.seg_net.model as network
from fastdlo.seg_net.dataset import BasicDataset


model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}

class SegNet():
    

    def __init__(self, model_name, checkpoint_path, img_w, img_h):

        self.model = model_map[model_name](num_classes=1, output_stride=16)
        network.convert_to_separable_conv(self.model.classifier)

        logging.info("Loading model {}".format(model_name))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device {self.device}')

        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)

        logging.info("Model loaded !")

        self.img_w = img_w
        self.img_h = img_h
  


    def predict_img(self, img):
        self.model.eval() 

        img = cv2.resize(img, (self.img_w, self.img_h))

        img = torch.from_numpy(BasicDataset.pre_process(np.array(img)))

        img = img.unsqueeze(0)
        img = img.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(img)

            probs = torch.sigmoid(output)
            probs = probs.squeeze(0).cpu()

            full_mask = probs.squeeze().cpu().numpy()


        result = full_mask / np.max(full_mask)
        result = (result * 255).astype(np.uint8)
        return result

 