import torch
import torchvision
from torchvision import transforms
import pandas as pd
import cv2
import math
import face_alignment
import numpy as np
import imageio
import matplotlib.pyplot as plt

#          'nose'白色, 'left_eye'黑色, 'right_eye', 'left_ear', 'right_ear',  
#          'left_shoulder',  'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
#          'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
#           Left -  Green
#           Right - Red
PtColorMap = [(255, 255, 255), (0, 0, 0), (0, 0, 0), (255, 255, 255),(255, 255, 255), 
            (0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0),
            (0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0), (0, 255, 0), (255, 0, 0)]

# 畫線 0 (nose)-3(LEar), 0-4, ...
PtPairLst = [[0,3], [0, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [11, 12], [12, 14], [14, 16]]

AnglePtLst = [[5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16] ]

#frame_count = 1

def detect_device(i=None):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
    return device

def detect_device_for_face_alignment(i=None):
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    if use_cuda:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cuda")
    else:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device="cpu")
    return fa


def load_model(device):
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    model.eval()
    return model

# keyPts = the 17 key points of this person (x1, y1, visiablity1), ...
# frame = the image to draw the skeleton
def draw_skeleton(frame_count, frame, keyPts):
    frame_Lst = [frame_count]
    for i in range(17):
        x = keyPts[i][0]
        y = keyPts[i][1]
        cv2.circle(frame, (x, y), radius=10, color=PtColorMap[i], thickness=10) 
        frame_Lst = frame_Lst + [x, y]
    
    # 多畫 2 個點: 5 (LShoulder)-6 (RS) 中間,  11(LHip)-12(RH) 中間
    x5_6, y5_6, visiablity1  = (keyPts[5] + keyPts[6])/2
    x5_6 = int(x5_6)
    y5_6 = int(y5_6)
    x11_12, y11_12, visiablity2  = (keyPts[11] + keyPts[12])/2
    x11_12 = int(x11_12)
    y11_12 = int(y11_12)
    cv2.circle(frame, (x5_6, y5_6), radius=10, color=(0, 0, 255), thickness=10) 
    cv2.circle(frame, (x11_12, y11_12), radius=10, color=(0, 0, 255), thickness=10) 
    
        # 畫線 LShoulder-RS 5-6, Left Arm: 5-7-9,  RArm 6-8-10
        # LHip-RHip 11-12, Left Leg: 11-13-15, RLeg: 12-14-16
        # Body 0- (5-6中間)-(11-12 中間) 
    for pointPair in PtPairLst:
        ptIdx1, ptIdx2 = pointPair
        cv2.line(frame, (keyPts[ptIdx1][0], keyPts[ptIdx1][1]), (keyPts[ptIdx2][0], keyPts[ptIdx2][1]), color=(255, 255, 255), thickness=10) 
    cv2.line(frame, (keyPts[0][0], keyPts[0][1]), (x5_6, y5_6), color=(255, 255, 255), thickness=10)
    cv2.line(frame, (x5_6, y5_6), (x11_12, y11_12), color=(255, 255, 255), thickness=10)
    return frame_Lst

# 畫此 frame 中所有 score > RecognitionThreshold 的 subject 之 skeletons, 讓使用者輸入 SubjectIdx_of_first_tracking_frame
# kp[person_idx] = the 17 key points of this person (x1, y1, visiablity1), ...
def draw_all_skeletons(frame_count,frame, kp):
    for person_idx in range(kp.shape[0]):
        frame_Lst = draw_skeleton(frame_count, frame, kp[person_idx])
        # 在 Left hip ptNo=11 上顯示 person_idx
        #cv2.putText(frame,str(person_idx), (kp[person_idx][11][0], kp[person_idx][11][1]), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 0), 15)
    return frame_Lst

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

# keyPts = the 17 key points of this person (x1, y1, visiablity1), ...
# AnglePtLst = [[5, 7, 9], [6, 8, 10], [11, 13, 15], [12, 14, 16] ]
def draw_angles(frame, keyPts, AnglePtLst):
    # 先顯示幾個重要 joint angle: 7, 8, 13, 14, (11,12) 
    # Left elbow: 5-7-9, RElbow: 6-8-10, Left knee: 11-13-15, RKnee: 12-14-16
    for key in keyPts:
        for p1, p2, p3 in AnglePtLst:
            angle = int(getAngle((key[p1][0], key[p1][1]),(key[p2][0], key[p2][1]), (key[p3][0], key[p3][1])))
            angle = min(angle, 360-angle)    # 手腳的角度不會大於 180
            cv2.putText(frame,str(angle), (key[p2][0], key[p2][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

def skelton_draw(frame_count , fname,device,model,path1,path2):
    columnLst = ["frameNo"]
    for i in range(1, 18):
        xs = "x" + str(i)
        ys = "y" + str(i)
        columnLst = columnLst + [xs, ys]
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(7))
    cap.release()
    #total_frames = 300
    
    vid = imageio.get_reader(fname,  'ffmpeg')
    img = vid.get_data(0)
    width,height = img.shape[0],img.shape[1]
    fps = vid.get_meta_data()['fps']
    #fourcc = cv2.VideoWriter_fourcc(*'avc1')
    #output_movie = cv2.VideoWriter(path1, fourcc, fps, (width*2, height))
    writer = imageio.get_writer(path1, fps=fps)
    file_Lst = []
    print('No. of frames = ', total_frames)
    while(frame_count < total_frames):
        frame = vid.get_data(frame_count)  # Capture frame-by-frame
        try:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            img0 = np.copy(frame)
            transform = transforms.Compose([transforms.ToTensor()]) # Defing PyTorch Transform
            img = transform(frame).to(device) # Apply the transform to the image
        
            pred = model([img]) # Pass the image to the model
            pred_score = pred[0]['scores'].cpu().detach().numpy()
            i = 0
            while i < pred_score.shape[0]:
                if (pred_score[i] < 0.9): # 找出 score > RecognitionThreshold 的最小 idx
                    break
                i = i + 1
            kp = pred[0]['keypoints'].cpu().detach().numpy()
            kp = kp[:i]  # 只取 score > RecognitionThreshold 的 subject
                    
            frame_Lst = draw_all_skeletons(frame_count,img0, kp)
            #draw_angles(frame, kp[:], AnglePtLst)
                    # write data
            file_Lst.append(frame_Lst)
        except:
            frame_Lst = [frame_count]
            for i in range(17):
                x = 0
                y = 0
                frame_Lst = frame_Lst + [x, y]
            file_Lst.append(frame_Lst)
        try:
            img1 = np.append(frame,img0, axis=1)#把 2 張 img 接起來
            #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) # Convert to RGB
            #output_movie.write(img1)
            writer.append_data(img1)
        except:
            c=0
        frame_count += 1
        print(frame_count , end = ",")
            
    writer.close()
    #print(file_Lst)
    df = pd.DataFrame(file_Lst, columns = columnLst)
    Outfname = path2
    df.to_csv(Outfname, index = False)
    
def read_csv(fname):
    df = pd.read_csv(fname)
    columns = df.columns
    alldata = [[x for x in range(len(df[columns[0]]))] for y in range(len(columns))] 
    
    i = 0
    for index in df.columns:
        for j in range(len(df[index])):
            alldata[i][j] = str(df[index][j])
        i+=1
    
    return columns,alldata


def generate_chart(csvname,path1,path2):
    coordinate = [['x10','y10'],['x11','y11'],['x16','y16'],['x17','y17']]
    pos = ['left_hand','right_hand','left_ankle','right_ankle']
    lineup = ['2','1','4','3']
    df = pd.read_csv(csvname)
    #line chart
    for index in range(len(coordinate)):
        posx = df[coordinate[index][0]].values
        posy = df[coordinate[index][1]].values
        plt.title('%s line chart'%(pos[index]))
        plt.plot(posx, label='x axis')
        plt.plot(posy, label='y axis')
        plt.xlabel("Frames")
        plt.ylabel("Coordinate")
        plt.savefig(path1 + '%s.jpg'%(lineup[index]), dpi=500)
        plt.close()
        
    #scatter chart
    for index in range(len(coordinate)):
        posx = df[coordinate[index][0]].values
        posy = df[coordinate[index][1]].values
        plt.title('%s scatter chart'%(pos[index]))
        plt.scatter(posx,posy)
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.savefig(path2 + '%s.jpg'%(lineup[index]), dpi=500)
        plt.close()
    
def facepoint(frame_count , fname,detector,predictor,device,model,path1,path2):
    columnLst = ["frameNo"]
    for i in range(1, 82):
        xs = "x" + str(i)
        ys = "y" + str(i)
        columnLst = columnLst + [xs, ys]
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(7))
    #total_frames = 50
    width  = int(cap.get(3)) # float
    height = int(cap.get(4)) # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    output_movie = cv2.VideoWriter(path1, fourcc, fps, (width, height))
    file_Lst = []
    print('No. of frames = ', total_frames)
    while(frame_count < total_frames):
        _, frame = cap.read()  # Capture frame-by-frame
        try:
            faces = detector(frame)
            frame_Lst = [frame_count]
            landmarks = predictor(frame, faces[0])
            
            
            lst = [i for i in range(81)] 
            xmax = 0
            ymax = 0
            xmin = images.shape[0]
            ymin = images.shape[1]
            
            for i in lst:
                if(landmarks.part(i).x > xmax):
                    xmax = landmarks.part(i).x
                if(landmarks.part(i).y > ymax):
                    ymax = landmarks.part(i).y
                if(landmarks.part(i).x < xmin):
                    xmin = landmarks.part(i).x
                if(landmarks.part(i).y < ymin):
                    ymin = landmarks.part(i).y
            
            for face in faces:
                landmarks = predictor(frame, face)
                for n in range(landmarks.num_parts):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 2, (255, 255, 0), 2)
                    cv2.putText(frame, str(n+1), (x, y),
                                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                        fontScale=0.4,
                                        color=(0, 255, 255))
                    frame_Lst = frame_Lst + [x, y]
            file_Lst.append(frame_Lst)
            
                    
            #劃一個正方形        
            #cv2.line(images, (xmax,ymax), (xmax,ymin), color=(255, 125, 38), thickness=1)
            #cv2.line(images, (xmax,ymin), (xmin,ymin), color=(255, 125, 38), thickness=1)
            #cv2.line(images, (xmin,ymin), (xmin,ymax), color=(255, 125, 38), thickness=1)
            #cv2.line(images, (xmin,ymax), (xmax,ymax), color=(255, 125, 38), thickness=1)
            #cv2.circle(images, (int((xmax + xmin)/2),int((ymax + ymin)/2)), radius=1, color=(0, 0, 255), thickness=3)
        except:
            frame_Lst = [frame_count]
            for i in range(81):
                x = 0
                y = 0
                frame_Lst = frame_Lst + [x, y]
            file_Lst.append(frame_Lst)
        frame_count += 1
        output_movie.write(frame)
        print(frame_count , end = ",")
            
    cap.release()
    #print(file_Lst)
    #print(columnLst)
    df = pd.DataFrame(file_Lst, columns = columnLst)
    Outfname = path2
    df.to_csv(Outfname, index = False)
   
def face_alignment_draw(frame_count , fname,fa,path1,path2):
    columnLst = ["frameNo"]
    for i in range(1, 69):
        xs = "x" + str(i)
        ys = "y" + str(i)
        columnLst = columnLst + [xs, ys]
    cap = cv2.VideoCapture(fname)
    total_frames = int(cap.get(7))
    cap.release()
    #total_frames = 50
    
    vid = imageio.get_reader(fname,  'ffmpeg')
    img = vid.get_data(0)
    width,height = img.shape[0],img.shape[1]
    fps = vid.get_meta_data()['fps']
    #fourcc = cv2.VideoWriter_fourcc(*'avc1')
    #output_movie = cv2.VideoWriter(path1, fourcc, fps, (width*2, height))
    writer = imageio.get_writer(path1, fps=fps)
    file_Lst = []
    print('No. of frames = ', total_frames)
    while(frame_count < total_frames):
        frame = vid.get_data(frame_count)  # Capture frame-by-frame
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
        #img = np.copy(frame)
        #preds = fa.get_landmarks(frame)
        try:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
            img = np.copy(frame)
            preds = fa.get_landmarks(frame)
            frame_Lst = [frame_count]
            for faceIdx in range(1):
                for keyPoint in range(preds[0].shape[0]):
                    x = preds[faceIdx][keyPoint][0]
                    y = preds[faceIdx][keyPoint][1]
                    cv2.circle(img, (x, y), radius=2, color=(255, 255, 255), thickness=3) 
                    frame_Lst = frame_Lst + [x, y]
            file_Lst.append(frame_Lst)
            
                    
            #劃一個正方形        
            #cv2.line(images, (xmax,ymax), (xmax,ymin), color=(255, 125, 38), thickness=1)
            #cv2.line(images, (xmax,ymin), (xmin,ymin), color=(255, 125, 38), thickness=1)
            #cv2.line(images, (xmin,ymin), (xmin,ymax), color=(255, 125, 38), thickness=1)
            #cv2.line(images, (xmin,ymax), (xmax,ymax), color=(255, 125, 38), thickness=1)
            #cv2.circle(images, (int((xmax + xmin)/2),int((ymax + ymin)/2)), radius=1, color=(0, 0, 255), thickness=3)
        except:
            frame_Lst = [frame_count]
            for i in range(68):
                x = 0
                y = 0
                frame_Lst = frame_Lst + [x, y]
            file_Lst.append(frame_Lst)
        try:
            img1 = np.append(frame, img, axis=1)#把 2 張 img 接起來
            #img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR) # Convert to RGB
            #output_movie.write(img1)
            writer.append_data(img1)
        except:
            c=0
        frame_count += 1
        
        print(frame_count , end = ",")
            
    writer.close()
    #print(file_Lst)
    #print(columnLst)
    df = pd.DataFrame(file_Lst, columns = columnLst)
    Outfname = path2
    df.to_csv(Outfname, index = False)







