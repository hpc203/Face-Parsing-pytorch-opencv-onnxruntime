from net import BiSeNet
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse

def vis_parsing_maps(im, parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_parsing_anno_color, vis_im

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Parse')
    parser.add_argument('--imgpath', default='116_ori.png', type=str, help='A path to an image to use for display.')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_classes = 19
    net = BiSeNet(n_classes)
    net.to(device)
    net.load_state_dict(torch.load('my_params.pth', map_location=device))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = Image.open(args.imgpath)
    with torch.no_grad():
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        np.save('torch_input.npy', img.cpu().numpy())
        out = net(img)
        np.save('torch_out.npy', out.cpu().numpy())
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    vis_parsing_anno_color, vis_im = vis_parsing_maps(image, parsing, stride=1)

    cv2.namedWindow('vis_parsing_anno_color', cv2.WINDOW_NORMAL)
    cv2.imshow('vis_parsing_anno_color', vis_parsing_anno_color)
    cv2.namedWindow('vis_im', cv2.WINDOW_NORMAL)
    cv2.imshow('vis_im', vis_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()