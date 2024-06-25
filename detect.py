from __future__ import print_function
import argparse
import threading

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision

from data import cfg_mnet, cfg_re50, cfg_mnetv3, cfg_gnet
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from utils.box_utils import decode, decode_landm
import time
from math import *
import os
import imutils
from matplotlib import pyplot as plt
from scipy.integrate import simps
from torch.autograd import Variable
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PFLD.utils import softmax_temperature
from PFLD.PFLD_MobileVIT import mobile_vit_small
from PFLD.util import calculate_pitch_yaw_roll, draw_axis
from scipy.spatial import distance as dist
from queue import Queue
from models.pfld import PFLDInference, AuxiliaryNet
from models.detector import detect_faces
cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
# torch.cuda.set_device(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# import get_label from '/public/MountData/DataDir/mye/mye/novamye/project1/mojing_detection/Ghost+VIT/deep-learning-for-image-processing-master/deep-learning-for-image-processing-master/pytorch_classification/vision_transformer'
# from vision_transformer.predict import get_label



import pygame
import time

event_queue = Queue()

def process_events():
    while True:
        event = event_queue.get()
        if event == "play_sound":
            play_warning_sound()
def play_warning_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("music/alarm.ogg")  # 替换为你的警告声音文件路径
    pygame.mixer.music.play()
    time.sleep(2)  # 播放三秒钟
    pygame.mixer.music.stop()

event_thread = threading.Thread(target=process_events)
event_thread.start()


def get_MER(x1, x2, y1, y2, img):
    cropped = img[y1:y2, x1:x2]  # 裁剪坐标为[y0:y1, x0:x1]
    return cropped


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def eye_aspect_ratio(eye):
    # 垂直眼标志（X，Y）坐标
    A = dist.euclidean(eye[1], eye[7])  # 计算两个集合之间的欧式距离
    B = dist.euclidean(eye[3], eye[5])
    D = dist.euclidean(eye[2], eye[6])
    # 计算水平之间的欧几里得距离
    # 水平眼标志（X，Y）坐标
    C = dist.euclidean(eye[0], eye[4])
    # 眼睛长宽比的计算
    ear = (A + B + D) / (3.0 * C)
    # 返回眼睛的长宽比
    return ear


def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])  # 89, 95
    B = np.linalg.norm(mouth[14] - mouth[18])  # 90, 94
    D = np.linalg.norm(mouth[15] - mouth[17])  # 91, 93
    C = np.linalg.norm(mouth[12] - mouth[16])  # 88, 92
    mar = (A + B + D) / (3.0 * C)
    return mar



def count_mouth(mouth_state):
    mouth_count = 0
    mouth_close = 0
    Roll_mouth = 0
    for i in range(len(mouth_state)):
        if mouth_state[i] == 'mouth_open':
            Roll_mouth += 1
            mouth_count += 1
        else:
            if mouth_count > 10:
                mouth_close += 1
            mouth_count = 0
    return Roll_mouth, mouth_close


def detect(img_raw):
    start = time.time()
    # 将图片转换为 PyTorch 张量，并移到 GPU 上（如果可用）
    img = torch.tensor(img_raw, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    # print('1',time.time() - start)
    # start = time.time()
    # 获取图片的高度和宽度
    im_height, im_width = img_raw.shape[:2]
    # print('2',time.time() - start)
    # start = time.time()
    # 计算图片的缩放因子
    scale = torch.tensor([im_width, im_height, im_width, im_height], dtype=torch.float32).to(device)
    # print('3',time.time() - start)
    # start = time.time()
    # 进行前向传播，获取预测结果
    loc, conf, landms = net(img)
    # print('4',time.time() - start)
    # start = time.time()
    # 根据每一个图片计算它对应的先验框
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    # print('5',time.time() - start)
    # start = time.time()
    prior_data = priorbox.forward().to(device).data
    # print('6',time.time() - start)
    # start = time.time()
    # 对先验框进行调整获得最终的预测框，为归一化的形式
    boxes = (decode(loc.data.squeeze(0), prior_data, cfg['variance'])* scale ).cpu().numpy()
    # print('6',time.time() - start)
    # start = time.time()
    # ignore low scores  and  keep top-K before NMS
    inds = np.where(conf.squeeze(0)[:, 1].cpu().numpy() > args.confidence_threshold)[0]
    # print('7',time.time() - start)
    # start = time.time()
    order = conf.squeeze(0)[:, 1].cpu().numpy()[inds].argsort()[::-1][:args.top_k]
    # print('8',time.time() - start)
    # start = time.time()

    dets = np.hstack((boxes[inds][order], conf.squeeze(0)[:, 1].cpu().numpy()[inds][order][:, np.newaxis])).astype(np.float32, copy=False)  # 目标框和分数框的堆叠
    # print('9',time.time() - start)
    # start = time.time()
    keep = py_cpu_nms(dets, args.nms_threshold)
    # print('10',time.time() - start)
    # start = time.time()
    dets = np.concatenate((dets[keep, :][:args.keep_top_k, :], (decode_landm(landms.data.squeeze(0), prior_data, cfg['variance']) * torch.Tensor([im_width, im_height, im_width, im_height,

                           im_width, im_height, im_width, im_height,
                           im_width, im_height]).to(device)).cpu().numpy()[inds][order][keep][:args.keep_top_k, :]), axis=1)   # 将数组dets和landms在第1轴上连接起来，并将结果存储在dets中
    # print('11',time.time() - start)
    # start = time.time()
    ear, mar, pitch, yaw, roll, image_size, euler_angles_landmark = 0, 0, 0, None, 0, 112, []

    print('第一段处理时间',time.time()-start)
    start = time.time()
    for b in dets:
        if b[4] < args.vis_thres:
            continue
        b = list(map(int, b))
        x1, y1, x2, y2 = b[0], b[1], b[2], b[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h]) * 1.1)  # 当前边界框尺寸的1.1倍，是个正方形

        cx = x1 + w // 2
        cy = y1 + h // 2
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        # 确保它们在图像的范围
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - im_width)
        edy = max(0, y2 - im_height)
        x2 = min(im_width, x2)
        y2 = min(im_height, y2)
        cropped = img_raw[y1:y2, x1:x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:  # 说明扩大的人脸矩形框超出了本身图像的范围
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT,
                                         0)  # 对裁剪区域进行边界填充操作， 并将填充像素值设置为0,确保裁剪后的感兴趣区域大小与原始图像一致
        cropped = cv2.resize(cropped, (image_size, image_size))
        cropped = np.asarray(cropped)
        input = np.asarray(cv2.resize(cropped, (image_size, image_size))).astype(float)
        img = np.expand_dims(input, 0)
        img = torch.from_numpy(img)
        img = img.float()
        img = img.to(device)
        pre_landmarks, _ = pfld_backbone(img)
        pre_landmarks = pre_landmarks.cpu().numpy()
        pre_landmarks = pre_landmarks.reshape(pre_landmarks.shape[0], -1, 2)

        pre_landmark = pre_landmarks[0, :, :2] * [112, 112]

        pre_landmark = pre_landmark * [size / image_size, size / image_size] - [dx, dy]
        # print(pre_landmark)
        pre = []
        for ptidx, (x, y) in enumerate(pre_landmark):
            cv2.circle(img_raw, (x1 + int(x), y1 + int(y)), 1, (0, 0, 255), -1) # 1是圆的半径，2是圆的宽
            cv2.circle(img_raw, (x1 + int(x), y1 + int(y)), 1, (0, 0, 255), 1)
            pre.append(x1 + int(x))
            pre.append(y1 + int(y))

        pre = (list(i for i in np.array(pre).reshape(-1, 2)))


        cv2.rectangle(img_raw, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # # landms
        cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 2)  # right
        cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 2)  # left
        cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 2)  # bizi
        cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 2)  # mouth_right
        cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 2)  # mouth_left

        # 第九步：提取左眼和右眼坐标

        leftEye = pre_landmarks[0, 60:68]
        rightEye = pre_landmarks[0, 68:76]
        # 嘴巴坐标
        mouth = pre_landmarks[0, 76:96]
        # 获取头部姿态
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for index in TRACKED_POINTS:

            euler_angles_landmark.append((pre[index]).tolist())
        euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
        euler_angle = calculate_pitch_yaw_roll(euler_angles_landmark[0])
        euler_angles_landmark = []

        # print('e:',euler_angle)
        pitch = format(euler_angle[0, 0])
        yaw = format(euler_angle[1, 0])
        roll = format(euler_angle[2, 0])

        # print('p,y,r:', pitch, yaw, roll)
        # img_raw = draw_axis(img_raw, img_raw[y1:y2, x1:x2, :], x1,y1, float(yaw), float(pitch), float(roll))
        # 第十步：构造函数计算左右眼的EAR值，使用平均值作为最终的EAR
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

    print('第二段处理时间',time.time()-start)

    return img_raw, ear, mar, pitch, yaw, roll


def get_file_list(file_path):
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list


def save_label(labels, label_path):
    with open(label_path, 'w') as f:
        for label in labels:
            f.write(label + '\n')


# file_path = './Male/'
# dir_list = get_file_list(file_path)



def CalculationIndex(coordinate):
# -------------------- 计算ear --------------------------
    leftEye, rightEye = coordinate[60:68], coordinate[68:76]
    leftEAR, rightEAR = eye_aspect_ratio(leftEye), eye_aspect_ratio(rightEye)
    ear = (leftEAR + rightEAR) / 2.0
# -------------------- 计算mar --------------------------
    mouth = coordinate[76:96]
    mar = mouth_aspect_ratio(mouth)
# -------------------- 计算pitch yaw roll --------------------------
    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    euler_angles_landmark = []
    for index in TRACKED_POINTS:
        euler_angles_landmark.append((coordinate[index]).tolist())
    euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
    euler_angle = calculate_pitch_yaw_roll(euler_angles_landmark[0])
    # print('e:',euler_angle)
    pitch = format(euler_angle[0, 0])
    yaw = format(euler_angle[1, 0])
    roll = format(euler_angle[2, 0])
    return ear, mar, pitch, yaw, roll

if __name__ == '__main__':
# ------------------------------------ 环境配置 ------------------------------------------
    T1 = time.perf_counter()
    parser = argparse.ArgumentParser(description='Retinaface')
    parser.add_argument('-m', '--trained_model', default='./weights/mobilev3_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='mobilev3',
                        help='Backbone network mobile0.25 & resnet50 & ghostnet & mobilev3')
    parser.add_argument('--image', type=str, default=r'./curve/face.jpg', help='detect images')
    parser.add_argument('--fourcc', type=int, default=1, help='detect on webcam')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    # parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--confidence_threshold', default=0.9, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.9, type=float, help='visualization_threshold')
    # parser.add_argument('--model_path', default="./weights/best_checkpoint_epoch_.pth.tar", type=str)
    parser.add_argument('--model_path', default="./weights/checkpoint_epoch_500.pth.tar")
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.set_grad_enabled(False)

    # cfg = cfg_mnetv3
    # if args.network == "mobile0.25":
    #     from models.retinaface_m import RetinaFace
    #
    #     cfg = cfg_mnet
    # elif args.network == "resnet50":
    #     from models.retinaface_m import RetinaFace
    #
    #     cfg = cfg_re50
    # elif args.network == "ghostnet":
    #     from models.retinaface_g import RetinaFace
    #
    #     cfg = cfg_gnet
    # elif args.network == "mobilev3":
    #     from models.retinaface_g import RetinaFace
    #
    #     cfg = cfg_mnetv3

# ------------------------------------ 网络和模型 ------------------------------------------
#     net = RetinaFace(cfg=cfg, phase='test')
#     net = load_model(net, args.trained_model, args.cpu)
#     net.eval()
#     print('Finished loading model!')
#     cudnn.benchmark = True
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = net.to(device)
#     pfld_backbone = mobile_vit_small()
    checkpoint = torch.load(args.model_path, map_location=device)
    pfld_backbone = PFLDInference().to(device)
    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    pfld_backbone = pfld_backbone.to(device)
    pfld_backbone.eval()
    pfld_backbone = pfld_backbone.to(device)
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()])

# ------------------------------------ 校验参数 ------------------------------------------
    EYE_AR_THRESH = 0.2  # 眼睛EAR阈值   5帧
    MAR_THRESH = 0.2   # 嘴巴EAR阈值    3帧
    POSTURE_PITCH_THRESH = 30  # 姿态角   3帧
    POSTURE_YAW_THRESH = 45
    COUNTER_EYE = 0
    COUNTER_MAR = 0
    COUNTER_POSTURE = 0
    TOTAL_EYE = 0
    TOTAL_MAR = 0
    TOTAL_POSTURE = 0
    eRoll = 0

# ------------------------------------ 开启摄像头 ------------------------------------------

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 300)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        x, y, w, h = 3,3,3,3
        try:
            x, y, w, h = faces[0]
        except:
            pass
        # 边界检查，确保裁剪区域不超出图像边界
        x1 = max(max(0, x) - 51, 3)
        y1 = max(max(0, y) - 51, 3)
        x2 = min(x + w, img.shape[1]) + 50
        y2 = min(y + h, img.shape[0]) + 50

        # 裁剪图像
        img = img[y1:y2, x1:x2]
        if not ret:
            break
        start = time.time()
        height, width = img.shape[:2]
        bounding_boxes, landmarks = detect_faces(img)
        for box in bounding_boxes:
            x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)

            w = x2 - x1 + 1
            h = y2 - y1 + 1
            cx = x1 + w // 2
            cy = y1 + h // 2

            size = int(max([w, h]) * 1.1)
            x1 = cx - size // 2
            x2 = x1 + size
            y1 = cy - size // 2
            y2 = y1 + size

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            edx1 = max(0, -x1)
            edy1 = max(0, -y1)
            edx2 = max(0, x2 - width)
            edy2 = max(0, y2 - height)

            cropped = img[y1:y2, x1:x2]
            if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
                cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                             cv2.BORDER_CONSTANT, 0)

            input = cv2.resize(cropped, (112, 112))
            input = transform(input).unsqueeze(0).to(device)
            _, landmarks = pfld_backbone(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(
                -1, 2) * [size, size] - [edx1, edy1]
            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255),2)
            if len(pre_landmark) > 0:
                ear, mar, pitch, yaw, roll = CalculationIndex(pre_landmark.astype(np.int32))
                cv2.putText(img, "ear: {:.2f}".format(ear), (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),1)
                cv2.putText(img, "mar: {:.2f}".format(mar), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                cv2.putText(img, "pitch: {}".format(pitch), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),1)
                cv2.putText(img, "yaw: {}".format(yaw), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),   1)
                cv2.putText(img, "roll: {}".format(roll), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
        ###########################    眼部疲劳检测     ###########################
                ear = ear if ear != 0 else 1
                if ear <= EYE_AR_THRESH:  # 眼睛长宽比：0.25
                    COUNTER_EYE += 1
                if ear > EYE_AR_THRESH:
                    COUNTER_EYE = 0
                if COUNTER_EYE >= 4:
                    print(COUNTER_EYE)
                    COUNTER_EYE = 0
                    event_queue.put("play_sound")
                    cv2.putText(img, "疲劳状态", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                    print("疲劳状态")

        ###########################   打哈欠检测     ###########################
                mar = mar if (mar) is not None else 5
                if mar >= MAR_THRESH:  #
                    COUNTER_MAR += 1
                if mar < MAR_THRESH:
                    COUNTER_MAR = 0
                if COUNTER_MAR >= 3:
                    COUNTER_MAR = 0
                    event_queue.put("play_sound")
                    cv2.putText(img, "打哈欠状态", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                    print("打哈欠状态")

        ###########################   姿态异常检测     ###########################
                pitch = pitch if pitch != 0 else 0
                yaw = yaw if yaw is not None else 0
                if abs(float(pitch)) >= 20 or abs(float(yaw)) >= 9:
                    COUNTER_POSTURE += 1
                if COUNTER_POSTURE >= 3:
                    COUNTER_POSTURE = 0
                    event_queue.put("play_sound")
                    cv2.putText(img, "姿态异常", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
                    print("姿态异常")
                else:
                    COUNTER_POSTURE = 0

        cv2.imshow('face_landmark_68', img)
        if cv2.waitKey(10) == 27:
            break

#         ret, img = cap.read()
#         # img = cv2.resize(img, (320, 240))  # 这里可以调整分辨率大小
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#         # 人脸检测
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         print(faces)
#         x, y, w, h = 3,3,3,3
#         try:
#             x, y, w, h = faces[0]
#         except:
#             pass
#         # 边界检查，确保裁剪区域不超出图像边界
#         x1 = max(max(0, x) - 51, 3)
#         y1 = max(max(0, y) - 51, 3)
#         x2 = min(x + w, img.shape[1]) + 50
#         y2 = min(y + h, img.shape[0]) + 50
#
#         # 裁剪图像
#         img = img[y1:y2, x1:x2]
#         if not ret:
#             break
#         start = time.time()
#         img_draw, ear, mar, pitch, yaw, roll = detect(img)
#         # print(time.time() - start)
#         # start = time.time()
#
#         cv2.putText(img_draw, "ear: {:.2f}".format(ear), (30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#         cv2.putText(img_draw, "mar: {:.2f}".format(mar), (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#         cv2.putText(img_draw, "pitch: {}".format(pitch), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),1)
#         cv2.putText(img_draw, "yaw: {}".format(yaw), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),   1)
#         cv2.putText(img_draw, "roll: {}".format(roll), (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#         # print(time.time() - start)
#
#
# ###########################    眼部疲劳检测     ###########################
#         ear = ear if ear != 0 else 1
#         if ear <= EYE_AR_THRESH:  # 眼睛长宽比：0.25
#             COUNTER_EYE += 1
#         if ear > EYE_AR_THRESH:
#             COUNTER_EYE = 0
#         if COUNTER_EYE >= 4:
#             print(COUNTER_EYE)
#             COUNTER_EYE = 0
#             event_queue.put("play_sound")
#             cv2.putText(img_draw, "疲劳状态", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#             print("疲劳状态")
#
#
#
#
# ###########################   打哈欠检测     ###########################
#         mar = mar if (mar) is not None else 5
#         if mar >= MAR_THRESH:  #
#             COUNTER_MAR += 1
#         if mar < MAR_THRESH:
#             COUNTER_MAR = 0
#         if COUNTER_MAR >= 3:
#             COUNTER_MAR = 0
#             event_queue.put("play_sound")
#             cv2.putText(img_draw, "打哈欠状态", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#             print("打哈欠状态")
#
#
#
# ###########################   姿态异常检测     ###########################
#         pitch = pitch if pitch != 0 else 0
#         yaw = yaw if yaw is not None else 0
#         if abs(float(pitch)) >= 20 or abs(float(yaw)) >= 9:
#             COUNTER_POSTURE += 1
#         if COUNTER_POSTURE >= 3:
#             COUNTER_POSTURE = 0
#             event_queue.put("play_sound")
#             cv2.putText(img_draw, "姿态异常", (30, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#             print("姿态异常")
#         else:
#             COUNTER_POSTURE = 0
#
#
#
#         #
#         #
#         # else:
#         #     # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
#         #     if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：5
#         #         TOTAL += 1
#         #         E.append(COUNTER)
#         #     # 重置眼帧计数器
#         #     COUNTER = 0
#         #     if ear < 0.20:  # 眼睛长宽比：0.2
#         #         eRoll += 1
#         cv2.imshow('face_landmark_98', img_draw)
#         if cv2.waitKey(10) == 27:
#             break
#     print("end")
#     #
#     # for dir in dir_list:
#     #     ext = os.path.splitext(dir)[1]  # 获取后缀名
#     #     if ext == '.mp4' or '.avi':
#     #         video_path = os.path.join(file_path, dir)
#     # video_path = '/public/MountData/DataDir/mye/mye/novamye/ConvNext_Retinaface/video/9-1 - Trim.mp4'
    # cap = cv2.VideoCapture(video_path)
#     cap = cv2.VideoCapture(
#         './video/video.mp4')
#     cap.set(3, 640)  # set video width
#     cap.set(4, 640)  # set video height
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     print('f',fps)
#     fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     # outVideo = cv2.VideoWriter('./video/save_video_kcf2.avi', fourcc, fps,
#     #         (720, 540))
#     # # # create VideoWriter for saving
#     outVideo = cv2.VideoWriter(
#         './video/videoSave.mp4', fourcc, fps,
#         (width, height))
#     biaoqian = 0
#     # 定义两个常数
#     # 眼睛长宽比
#     # 闪烁阈值
#     EYE_AR_THRESH = 0.22
#     EYE_AR_CONSEC_FRAMES = 5
#     # 打哈欠长宽比
#     # 闪烁阈值
#     MAR_THRESH = 0.6
#     MOUTH_AR_CONSEC_FRAMES = 90
#     # 初始化帧计数器和眨眼总数
#     COUNTER = 0
#     TOTAL = 0  # fe1
#     # 初始化帧计数器和打哈欠总数
#     mCOUNTER = 0
#     mTOTAL = 0
#
#     eRoll = 0  # fe3
#     mRoll = 0  # fm
#     Roll = 0  # 总帧数
#     E = []  # fe2
#     M = []
#     P = []
#     Y = []
#     A = []
#     B = []
#     pCOUNTER = 0
#     yCOUNTER = 0
#     pRoll = 0
#     yRoll = 0
#     frame_index = 0
#     # path1 = './video/copy/'
#     # path2 = './video/copy2/'
#     # path3 = './video/frame/'
#     cishu = 0
#     # while True:
#     print(2)
#     print(biaoqian)
#     ret, frame = cap.read()
#     print(ret)
#     biaoqian += 1
#     # img = cv2.imread(frame)
#     # if ret is None:
#     #     break
#     # if frame is None:
#     #     break
#     frame = imutils.resize(frame, width=720)
#     if frame_index == 1763:
#         cv2.imwrite('./video/1763_3.png', frame)
#     frame_index += 1
#     print(frame_index)
#     print(frame)
#     save_folder = './video/'
#     save_name = os.path.join(save_folder, "data")  # 添加文件扩展名
#     dirname = os.path.dirname(save_name)
#     if not os.path.isdir(dirname):
#         os.makedirs(dirname)
#     with open(save_name + 'data.txt', "wb") as fd:
#         file_name = os.path.basename(save_name)[:-4] + "\n"
#         print('sb',file_name)
#         # fd.write(file_name)
#         while True:
#             ret, frame = cap.read()
#             if ret is None:
#                 break
#             if frame is None:
#                 break
#             if ret:
#                 copimg, copimg2, img_raw, ear, mar, pitch, yaw, roll = detect(frame)
#                 outVideo.write(img_raw)
#         line = str(mar) + " \n"
#         fd.write(line)
#         frame = cv2.imread('./imgs/3.jpg')
#         copimg, copimg2, img_raw, ear, mar, pitch, yaw, roll = detect(frame)
#         print(mar)
#         cv2.imwrite('./figueRes.jpg', img_raw)
#
# #第十三步：循环，满足条件的，眨眼次数+1
#     if ear < EYE_AR_THRESH:  # 眼睛长宽比：0.25
#         COUNTER += 1
#     else:
#         # 如果连续3次都小于阈值，则表示进行了一次眨眼活动
#         if COUNTER >= EYE_AR_CONSEC_FRAMES:  # 阈值：3
#             TOTAL += 1
#             E.append(COUNTER)
#         # 重置眼帧计数器
#         COUNTER = 0
#
#     if ear < 0.20:  # 眼睛长宽比：0.2
#         eRoll += 1
#
#
#     # 同理，判断是否打哈欠
#     if mar > MAR_THRESH:  # 张嘴阈值0.6
#         mCOUNTER += 1
#         mRoll += 1
#     else:
#         # 如果连续3次都小于阈值，则表示打了一次哈欠
#         if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:  # 阈值：3
#             M.append(mCOUNTER)
#             mTOTAL += 1
#         # 重置嘴帧计数器
#         mCOUNTER = 0
#
#     pitch = float(pitch)
#     yaw = float(yaw)
#     roll = float(roll)
#     if abs(pitch) >= 30:
#         pCOUNTER += 1
#         pRoll += 1
#     else:
#         P.append(pCOUNTER)
#         pCOUNTER = 0
#
#     if abs(yaw) >= 45:
#         yCOUNTER += 1
#         yRoll += 1
#     else:
#         Y.append(yCOUNTER)
#         yCOUNTER = 0
#
#     Roll += 1
#
#     if (Roll / round(fps)) % 2 == 0:
#         A.append(eRoll/Roll)
#         eRoll = 0
#
#
#     if (Roll / round(fps)) % 10 == 0:
#         B.append(mRoll/Roll)
#         mRoll = 0
#
#
#     if (Roll / round(fps)) % 30 == 0:
#
#         cishu += 1
#         print('A',A)
#         print('A',B)
#         print('E', E)
#         print('M', M)
#         print('P', P)
#         print('Y', Y)
#         if not A:
#             a = 0
#         else:
#             a = max(A)
#
#         if not B:
#             b = 0
#         else:
#             b = max(B)
#
#         if not E:
#             e = 0
#         else:
#             e = max(E) / fps
#
#         if not M:
#             m = 0
#         else:
#             m = max(M) / fps
#
#         if not P:
#             p = 0
#         else:
#             p = max(P) / fps
#
#         if not Y:
#             y = 0
#         else:
#             y = max(Y) / fps
#         # fe1 = TOTAL
#         Fe1 = a
#         Fe2 = e
#         Fm1 = b
#         Fm2 = m
#         Fp = p
#         Fy = y
#
#         if (Fe1 >= 0.4 or Fe2 >= 2 or Fm1 >= 0.3 or Fm2 >= 3 or Fp >= 3 or Fy >= 3):
#             # cv2.putText(img_raw, "state: fatigue", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
#             pilao += 1
#             print('次数：',cishu)
#             print("疲劳驾驶")
#             print('pilao:',pilao)
# # # 画图操作
# # cv2.putText(img_raw, "fe1: {:.2f}".format(fe1), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "fe2: {:.2f}".format(fe2), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "fe3: {:.2f}".format(fe3), (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "fm1: {:.2f}".format(fm), (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "fm2: {:.2f}".format(fm2), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "pitch: " + "{:.2f}".format(pitch), (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),1)
# # cv2.putText(img_raw, "yaw: " + "{:.2f}".format(yaw), (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),   1)
# # cv2.putText(img_raw, "roll: " + "{:.2f}".format(roll), (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# #
# # cv2.putText(img_raw, "Fe1: 2.57", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "Fe2: closed eye", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "Fy1: 3.34", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "Fy2: yawning", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.putText(img_raw, "state:fatigue", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# # cv2.imwrite('./figueRes.jpg', img_raw)
# #
# # if  (fe1 > 15 or 0 < fe1 < 10 or fe2 > 2 or fe3 >= 0.4 or fm >= 0.3 or fm2 > 2 or fp > 2 or fy > 2):
# #     cv2.putText(img_raw, "state: fatigue", (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 1)
# #     print("疲劳驾驶")
# #     pilao += 1
# # print("项目疲劳的准确度为：",len(dir_list) / pilao)
# #
# # outVideo.write(img_raw)
# #
# # output_path = f"{output_folder}/frame_{frame_index}.jpg"
# #
# # 保存当前帧为图像文件
# # cv2.imwrite(path1 + str(frame_index) + '.jpg', copimg)
# # cv2.imwrite(path2 + str(frame_index) + '.jpg', copimg2)
# # cv2.imwrite(path3 + str(frame_index) + '.jpg', img_raw)
