import cv2
import json
import numpy as np
import os
import time
import glob

from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    phi = 0
    weighted_bifpn = True
    model_path = 'efficientdet-d0.h5'
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    # coco classes
    classes = {value['id'] - 1: value['name'] for value in json.load(open('coco_90.json', 'r')).values()}
    num_classes = 90
    score_threshold = 0.5
    colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
    

    # _, model = efficientdet(phi=phi,
    #                         weighted_bifpn=weighted_bifpn,
    #                         num_classes=num_classes,
    #                         score_threshold=score_threshold)
    # model.load_weights(model_path, by_name=True)

    model = load_model('savedmodel/')

    video_path = 'test/video.mp4'
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./out.mp4', codec, fps, (width, height))
    times = []
    
    while True:
        ret, image = cap.read()
        if not ret:
            break
        src_image = image.copy()
        # BGR -> RGB
        image = image[:, :, ::-1]
        h, w = image.shape[:2]

        image, scale = preprocess_image(image, image_size=image_size)
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
        end = time.time()
        print(end - start)
        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        #calculating fps
        times.append(end-start)
        times = times[-20:]
        ms = sum(times)/len(times)*1000
        fps = 1000 / ms


        # select indices which have a score above the threshold
        indices = np.where(scores[:] > score_threshold)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]

        draw_boxes(src_image, boxes, scores, labels, colors, classes)
        src_image = cv2.putText(src_image, "Time: {:.1f}FPS".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        out.write(src_image)

        # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        # cv2.imshow('image', src_image)
        # cv2.waitKey(0)


if __name__ == '__main__':
    main()
