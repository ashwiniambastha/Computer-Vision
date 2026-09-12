"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw
import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from src.core import YAMLConfig
from src.core.yaml_config import COCOLabels


def process_image(image_path, model, labels_cfg):
    im_pil = Image.open(image_path).convert('RGB')
    process_and_save(im_pil, model, image_path, labels_cfg=labels_cfg)


def process_video(video_path, model, labels_cfg):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_image = process_and_save(im_pil, model, frame_num, save_img=False, labels_cfg=labels_cfg)
        
        frame_with_boxes = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        out.write(frame_with_boxes)

        # Show the frame with boxes
        cv2.imshow('Video Processing', frame_with_boxes)

        # Exit visualization on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_num += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_webcam(model, labels_cfg):
    cap = cv2.VideoCapture(0)
    frame_num = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('webcam_output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        processed_image = process_and_save(im_pil, model, frame_num, save_img=False, labels_cfg=labels_cfg)

        frame_with_boxes = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        out.write(frame_with_boxes)

        # Show the frame with boxes

        frame_num += 1
        
        cv2.imshow('Webcam', frame_with_boxes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_and_save(im_pil, model, identifier, save_img=True, labels_cfg=None):
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores, thrh=0.6, labels_cfg=labels_cfg)
    if save_img:
        identifier = os.path.splitext(identifier)[-2]
        identifier = identifier.split("/")[-1]
        # identifier = os.path.splitext(identifier)[-2]
        im_pil.save(f'result_{identifier}.jpg')
    
    return im_pil


def draw(images, labels, boxes, scores, thrh=0.7, labels_cfg=None):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            draw.rectangle(list(b), outline='red')
            draw.text((b[0], b[1]), text=f"{labels_cfg.get_label(lab[j].item())} {round(scrs[j].item(), 2)}", fill='blue')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)
    labels_cfg = COCOLabels(args.labels_file)
    labels_cfg.load_labels()

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    if args.input_type == 'image':
        process_image(args.file, model, labels_cfg)
    elif args.input_type == 'video':
        process_video(args.file, model, labels_cfg)
    elif args.input_type == 'webcam':
        process_webcam(model, labels_cfg)
    else:
        print("Invalid input type. Choose 'image', 'video', or 'webcam'.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--file', type=str, )
    parser.add_argument('-t', '--input_type', type=str,)
    parser.add_argument('-l', '--labels_file', type=str,)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
