import argparse
import time
from thop import profile
from thop import clever_format
from torchstat import stat

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps
from torch.autograd import Variable
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.datasets import WLFWDatasets
from utils.utils import softmax_temperature
from models.PFLD import PFLD
from models.PFLD_GhostNet import PFLD_GhostNet
from models.PFLD_GhostNet_Slim import PFLD_GhostNet_Slim
from models.PFLD_GhostOne import PFLD_GhostOne
from models.PFLD_MobileVIT import mobile_vit_xx_small,mobile_vit_x_small,mobile_vit_small,PFLD_mobileVIT_AuxiliaryNet
cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
torch.cuda.set_device(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark
    """
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i,], target[i,]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8,] - pts_gt[9,])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36,] - pts_gt[45,])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60,] - pts_gt[72,])
        else:
            print(L)
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


def compute_auc(errors, failureThreshold, step=0.0001, showCurve=True):
    nErrors = len(errors)
    xAxis = list(np.arange(0., failureThreshold + step, step))
    ced = [float(np.count_nonzero([errors <= x])) / nErrors for x in xAxis]

    AUC = simps(ced, x=xAxis) / failureThreshold
    failureRate = 1. - ced[-1]

    if showCurve:
        plt.plot(xAxis, ced)
        plt.show()

    return AUC, failureRate


def validate(pfld_backbone, val_dataloader, args):
    pfld_backbone.eval()
    idx_tensor = [idx for idx in range(121)]
    # print(idx_tensor)
    idx_tensor = Variable(torch.FloatTensor(idx_tensor)).to(device)
    nme_list = []
    cost_time = []
    mae = []
    yaw_error = .0
    pitch_error = .0
    roll_error = .0
    total = 0
    losses = []
    error = 0
    nme_list2 = []
    list_1, list_2, list_3, list_4, list_5, list_6 = [], [], [], [], [], []
    with torch.no_grad():
        idx = 0
        for img, landmark_gt,attribute_gt,euler_angle_gt,labels, cont_labels  in val_dataloader:


            img = img.float()
            img = img.to(device)
            # attribute_gt = attribute_gt.to(device)
            euler_angle_gt = euler_angle_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            start_time = time.time()
            landmark_pred,_ = pfld_backbone(img)
            cost_time.append(time.time() - start_time)
            landmark_pred = landmark_pred.reshape(landmark_pred.shape[0], -1, 2).cpu().numpy()
            landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            # print('euler_angle_gt:',euler_angle_gt)
            # print('angle:',angle)
            # print('torch.abs(euler_angle_gt - angle)',torch.abs(euler_angle_gt - angle))
            # error += torch.sum(torch.abs(euler_angle_gt - angle))
            # print('error:',error)
            # yaw, pitch, roll = auxiliarynet(x2)
            # # print('yaw:',yaw.shape)
            # # print('yaw:', yaw)

            #
            # landmarks = landmarks.cpu().numpy()
            # landmarks = landmarks.reshape(landmarks.shape[0], -1, 2)
            # landmark_gt = landmark_gt.reshape(landmark_gt.shape[0], -1, 2).cpu().numpy()
            #
            # label_yaw = cont_labels[:, 0].float()
            # label_pitch = cont_labels[:, 1].float()
            # label_roll = cont_labels[:, 2].float()
            # # print('lable:',label_yaw.shape)
            # # print('lable:', label_yaw)
            #
            #
            # # Binned predictions  预测属于哪一个bin，torch.Size([32])
            # _, yaw_bpred = torch.max(yaw.data, 1)
            # _, pitch_bpred = torch.max(pitch.data, 1)
            # _, roll_bpred = torch.max(roll.data, 1)
            # # print('yaw_bpred:',yaw_bpred.shape)
            # # print('yaw_bpred:', yaw_bpred)
            # # Continuous predictions
            # yaw_predicted = softmax_temperature(yaw.data, 1) # 归一化 softmax后的概率分布   torch.Size([32, 121])
            # pitch_predicted = softmax_temperature(pitch.data, 1)
            # roll_predicted = softmax_temperature(roll.data, 1)
            # # print('yaw_predicted:',yaw_predicted.shape)
            # # print('yaw_predicted:', yaw_predicted)
            # yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu() * 3 - 183   # torch.Size([32]) ,预测得到的角度
            # pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu() * 3 - 183
            # roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu() * 3 - 183
            # # print('yaw_predicted1:',yaw_predicted.shape)
            # # print('yaw_predicted1:',yaw_predicted)
            # # Mean absolute error
            # yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw))
            # pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch))
            # roll_error += torch.sum(torch.abs(roll_predicted - label_roll))
            #
            # if args.show_image:
            #     show_img = np.array(np.transpose((img[0] * 0.5 + 0.5).cpu().numpy(), (1, 2, 0)))
            #     show_img = (show_img * 255).astype(np.uint8)
            #     np.clip(show_img, 0, 255)
            #
            #     pre_landmark = landmarks[0, :, :2] * [args.input_size, args.input_size]
            #
            #     cv2.imwrite("xxx.jpg", show_img)
            #     img_clone = cv2.imread("xxx.jpg")
            #
            #     for ptidx, (x, y) in enumerate(pre_landmark):
            #         cv2.circle(img_clone, (int(x), int(y)), 1, (0, 0, 255), -1)
            #     cv2.imwrite("xx_{}.jpg".format(idx), img_clone)

            nme_temp = compute_nme(landmark_pred, landmark_gt[:, :, :2])

            # print(euler_angle_gt,angle)
            # pose_matrix = np.mean(np.abs(euler_angle_gt - angle), axis =0)
            # print('pose:',pose_matrix)
            # MAE = np.mean(pose_matrix)
            # mae.append(MAE)
            for item in nme_temp:
                # print('i:',item)
                for val in attribute_gt.numpy():
                    # print('val:',val)
                    if all(elem == 0 for elem in val):
                        nme_list2.append(item)
                    else:
                        # print("有非0元素")
                        zero_indexes = [i for i, val in enumerate(val) if val == 1]
                        # print(zero_indexes)
                        for zero_indexes in zero_indexes:
                            if zero_indexes == 0:
                                list_1.append(item)
                                # print('list1',list_1)
                            elif zero_indexes == 1:
                                list_2.append(item)
                                # print('list2', list_2)
                            elif zero_indexes == 2:
                                list_3.append(item)
                                # print('list3', list_3)
                            elif zero_indexes == 3:
                                list_4.append(item)
                                # print('list4', list_4)
                            elif zero_indexes == 4:
                                list_5.append(item)
                                # print('list5', list_5)
                            elif zero_indexes == 5:
                                list_6.append(item)
                                # print('list6', list_6)
                nme_list.append(item)
            idx += 1

        # nme

        # print('Test error in degrees of the model on the ' + str(total) +
        #       ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / total, pitch_error / total, roll_error / total))
        print('nme: {:.4f}'.format(np.mean(nme_list)))
        print('nme1: {:.4f}'.format(np.mean(list_1)))
        print('nme2: {:.4f}'.format(np.mean(list_2)))
        print('nme3: {:.4f}'.format(np.mean(list_3)))
        print('nme4: {:.4f}'.format(np.mean(list_4)))
        print('nme5: {:.4f}'.format(np.mean(list_5)))
        print('nme6: {:.4f}'.format(np.mean(list_6)))
        # print('mae: {:.4f}'.format(np.mean(MAE)))
        # auc and failure rate
        failureThreshold = 0.1
        auc, failure_rate = compute_auc(nme_list, failureThreshold)
        print('auc @ {:.1f} failureThreshold: {:.4f}'.format(failureThreshold, auc))
        print('failure_rate: {:}'.format(failure_rate))
        # inference time
        print("inference_cost_time: {0:4f}".format(np.mean(cost_time)))


def main(args):
    MODEL_DICT = {'PFLD': PFLD,
                  'PFLD_GhostNet': PFLD_GhostNet,
                  'PFLD_GhostNet_Slim': PFLD_GhostNet_Slim,
                  'PFLD_GhostOne': PFLD_GhostOne,
                  'mobile_vit_xx_small': mobile_vit_xx_small,
                  'mobile_vit_x_small': mobile_vit_x_small,
                  'mobile_vit_small': mobile_vit_small,
                  }
    MODEL_TYPE = args.model_type
    WIDTH_FACTOR = args.width_factor
    INPUT_SIZE = args.input_size
    LANDMARK_NUMBER = args.landmark_number
    NUM_Bins = 121
    # pfld_backbone = mobile_vit_small().to(device)  # 主干网络
    pfld_backbone = mobile_vit_small()  # 主干网络
    auxiliarynet = PFLD_mobileVIT_AuxiliaryNet().to(device)  # 辅助网络
    # print(MODEL_TYPE)
    checkpoint = torch.load(args.model_path, map_location=device)
    # print('c:',checkpoint)

    pfld_backbone.load_state_dict(checkpoint['pfld_backbone'])
    # auxiliarynet.load_state_dict(checkpoint["auxiliarynet"])
    input = torch.randn(1, 112, 112, 3)
    input = input.cuda()
    stat(pfld_backbone, (112, 112, 3))
    total = sum([param.nelement() for param in pfld_backbone.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    # macs, params = profile(pfld_backbone, inputs=(input,))
    # macs, params = clever_format([macs, params], "%.3f")  # 格式化输出
    # print('macs:{}'.format(macs))
    # print('params:{}'.format(params))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    # print(len(wlfw_val_dataset))
    wlfw_val_dataloader = DataLoader(wlfw_val_dataset, batch_size=1, shuffle=False, num_workers=8)

    # validate(pfld_backbone, wlfw_val_dataloader, args)



def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_type', default='mobile_vit_small', type=str)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--width_factor', default=1, type=float)
    parser.add_argument('--landmark_number', default=98, type=int)
    parser.add_argument('--device', default='gpu', type=str)
    parser.add_argument('--model_path', default="/public/MountData/DataDir/mye/mye/novamye/ConvNext_Retinaface/PFLD-pytorch-master/PFLD_GhostOne-main/checkpoint/snapshot/mobilevit1_wing_wn_an/best_checkpoint_epoch_.pth.tar", type=str)
    parser.add_argument('--test_dataset', default='/public/MountData/DataDir/mye/mye/novamye/ConvNext_Retinaface/PFLD-pytorch-master/data/test_data/list.txt', type=str)
    parser.add_argument('--show_image', default=False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
