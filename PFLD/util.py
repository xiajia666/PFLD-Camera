import cv2
import numpy as np
from math import cos, sin
import math

def calculate_pitch_yaw_roll(landmarks_2D,
                             cam_w=256,
                             cam_h=256,
                             radians=False):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """

    assert landmarks_2D is not None, 'landmarks_2D is None'

    # # Estimated camera matrix values.
    # c_x = cam_w / 2
    # c_y = cam_h / 2
    # f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    # f_y = f_x
    # camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
    #                             [0.0, 0.0, 1.0]])
    # print('c:',camera_matrix)
    # camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    # 相机坐标系(XYZ)：添加相机内参
    K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
         0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
         0.0, 0.0, 1.0]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
    # 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
    D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

    # 像素坐标系(xy)：填写凸轮的本征和畸变系数
    camera_matrix = np.array(K).reshape(3, 3).astype(np.float32)
    camera_distortion = np.array(D).reshape(5, 1).astype(np.float32)

    # dlib (68 landmark) trached points
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    # wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    # X-Y-Z with X pointing forward and Y on the left and Z up.
    # The X-Y-Z coordinates used are like the standard coordinates of ROS (robotic operative system)
    # OpenCV uses the reference usually used in computer vision:
    # X points to the right, Y down, Z to the front
    # landmarks_3D = np.float32([
    #     [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT,
    #     [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT,
    #     [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
    #     [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
    #     [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
    #     [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
    #     [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
    #     [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
    #     [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
    #     [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
    #     [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
    #     [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
    #     [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
    #     [0.000000, -7.415691, 4.070434],  # CHIN
    # ])

    landmarks_3D = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
                             [1.330353, 7.122144, 6.903745],  # 29左眉右角
                             [-1.330353, 7.122144, 6.903745],  # 34右眉左角
                             [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
                             [5.311432, 5.485328, 3.987654],  # 13左眼左上角
                             [1.789930, 5.393625, 4.413414],  # 17左眼右上角
                             [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
                             [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
                             [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
                             [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
                             [2.774015, -2.080775, 5.048531],  # 43嘴左上角
                             [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
                             [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
                             [0.000000, -7.415691, 4.070434]])  # 6下巴角

    # landmarks_3D = np.float32([[6.825897, 6.760612, 4.402142],  # 33左眉左上角
    #                          [1.330353, 7.122144, 6.903745],  # 29左眉右角
    #                          [-1.330353, 7.122144, 6.903745],  # 34右眉左角
    #                          [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
    #                          [5.311432, 5.485328, 3.987654],  # 13左眼左上角
    #                          [1.789930, 5.393625, 4.413414],  # 17左眼右上角
    #                          [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
    #                          [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
    #                          [7.308957, 0.913869, 0.000000],  # 12耳朵左角
    #                          [-7.308957, 0.913869, 0.000000],  # 0耳朵右角
    #                          [5.665918, -3.286078,1.022951],  # 10 脸部轮廓左
    #                          [-5.665918,-3.286078,1.022951],  # 2脸部轮廓右
    #                          [0.000000, -7.415691, 4.070434]])  # 6下巴角


    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    # print('landmarks_2D',landmarks_2D)
    # Applying the PnP solver to find the 3D pose of the head from the 2D position of the landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings points from the world coordinate system to the camera coordinate system.
    # 输出旋转矢量，与 tvec 一起将点从世界坐标系带到相机坐标系。
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    # rvec是旋转矩阵，tvec是平移矩阵，camera_matrix与K矩阵对应，camera_distortion与D矩阵对应。

    # print('image_pts:', landmarks_2D)
    # print('cam_matrix:',camera_matrix)
    # print('dist_coeffs:',camera_distortion)
    # print('object_pts:',landmarks_3D)
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)
    # Get as input the rotational vector, Return a rotational matrix

    # const double PI = 3.141592653;
    # double thetaz = atan2(r21, r11) / PI * 180;
    # double thetay = atan2(-1 * r31, sqrt(r32*r32 + r33*r33)) / PI * 180;
    # double thetax = atan2(r32, r33) / PI * 180;
    # print('rotation_vec:',rvec)
    # print('translation_vec:',tvec)
    rmat, _ = cv2.Rodrigues(rvec) # 输入src：旋转向量（3*1或者1*3）或者旋转矩阵（3*3）；输出dst：旋转矩阵（3*3）或者旋转向量（3*1或者1*3）；
    # print('rotation_mat:',rmat)
    pose_mat = cv2.hconcat((rmat, tvec))
    # print('pose_mat:', pose_mat)
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    # print('euler_angle:',euler_angles)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    # print('pitch, yaw, roll:',pitch, yaw, roll)
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    # print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    # print('euler_angles',euler_angles)
    return euler_angles  # euler_angles contain (pitch, yaw, roll)


def draw_axis(img, imgs, x,y,yaw, pitch, roll, tdx=None, tdy=None, size=50):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = imgs.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx + x
    y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                 * sin(pitch) * sin(yaw)) + tdy + y

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx + x
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                 * sin(yaw) * sin(roll)) + tdy + y

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx + x
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy + y

    cv2.line(img, (x+int(tdx), y+int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (x+int(tdx), y+int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (x+int(tdx), y+int(tdy)), (int(x3), int(y3)), (255, 0, 0), 3)

    return img




class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def vis_landmark(img_path, annotation, norm, point_num):
    """
    line format: [img_name bbox_x1 bbox_y1  bbox_x2 bbox_y2 landmark_x1 landmark y1 ...]
    """
    # check point len
    assert len(line) == 1 + 4 + point_num * 2  # img_path + bbox + point_num*2

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    img_name = annotation[0]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = annotation[1:5]
    landmark = annotation[5:]

    landmark_x = line[1 + 4::2]
    landmark_y = line[1 + 4 + 1::2]
    if norm:
        for i in range(len(landmark_x)):
            landmark_x[i] = landmark_x[i] * w
            landmark_y[i] = landmark_y[i] * h

    # draw bbox and face landmark
    cv2.rectangle(img, (int(bbox_x1), int(bbox_y1)),
                  (int(bbox_x2), int(bbox_y2)), (0, 0, 255), 2)
    for i in range(len(landmark_x)):
        cv2.circle(img, (int(landmark_x[i]), int(landmark_y[i])), 2,
                   (255, 0, 0), -1)

    cv2.imshow("image", img)
    cv2.waitKey(0)