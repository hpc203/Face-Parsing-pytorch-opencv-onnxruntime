import argparse
import cv2
import numpy as np

class face_parse:
    def __init__(self):
        self.net = cv2.dnn.readNet('my_param.onnx')
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        self.part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                       [255, 0, 85], [255, 0, 170],
                       [0, 255, 0], [85, 255, 0], [170, 255, 0],
                       [0, 255, 85], [0, 255, 170],
                       [0, 0, 255], [85, 0, 255], [170, 0, 255],
                       [0, 85, 255], [0, 170, 255],
                       [255, 255, 0], [255, 255, 85], [255, 255, 170],
                       [255, 0, 255], [255, 85, 255], [255, 170, 255],
                       [0, 255, 255], [85, 255, 255], [170, 255, 255]]
    def vis_parsing_maps(self, parsing_anno, stride):
        vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)
        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            vis_parsing_anno_color[index[0], index[1], :] = self.part_colors[pi]
        vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
        return vis_parsing_anno_color
    def parse(self, srcimg):
        img = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img /= 255.0
        img = (img - self.mean) / self.std
        blob = cv2.dnn.blobFromImage(img)
        np.save('cv_input.npy', blob)
        self.net.setInput(blob)
        out = self.net.forward()
        np.save('cv_out.npy', out)
        parsing = out.squeeze(0).argmax(0)
        vis_parsing_anno_color = self.vis_parsing_maps(parsing, stride=1)
        return vis_parsing_anno_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Parse')
    parser.add_argument('--imgpath', default='116_ori.png', type=str, help='A path to an image to use for display.')
    args = parser.parse_args()

    model = face_parse()
    srcimg = cv2.imread(args.imgpath)
    vis_parsing_anno_color = model.parse(srcimg)
    vis_parsing_anno_color = cv2.cvtColor(vis_parsing_anno_color, cv2.COLOR_RGB2BGR)
    cv2.namedWindow('vis_parsing_anno_color', cv2.WINDOW_NORMAL)
    cv2.imshow('vis_parsing_anno_color', vis_parsing_anno_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()