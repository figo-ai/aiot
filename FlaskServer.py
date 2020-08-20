from flask import Flask, render_template, request, redirect, url_for ,jsonify ,session 
from werkzeug.utils import secure_filename
import os
import uuid
from flask_bootstrap import Bootstrap
from function import *
import time

device = detect_device()
model = load_model(device)
#detector = dlib.get_frontal_face_detector()
#predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
# fa = detect_device_for_face_alignment()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
Bootstrap(app)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route('/', methods=['GET', 'POST'])
def home(name=None):
    return render_template("Test.html")

@app.route('/upload', methods=['POST'])
def uploadFile():
    # 选择的选项
    global options,filename,original_video,processed_video,excel_data,total_frames,columnLst,writer,file_Lst,vid,show_way,line_chart,scatter_chart,target5dir,target6dir,fname
    options = request.form.get('options')
    print('选择的选项是: '+ options)
    show_way = request.form.get('show')
    print('show_way选择的选项是: '+ show_way)
    # 处理档案
    #f = request.files['file']
    for upload in request.files.getlist("file"):
        f = upload
    print('上传的档案是: '+ f.filename)
    if(options == '1'):
        print('關節點')
        target = os.path.join(APP_ROOT, 'static/')
        target1 = os.path.join(APP_ROOT, 'static/joint/original/')
        target2 = os.path.join(APP_ROOT, 'static/joint/processed/')
        target3 = os.path.join(APP_ROOT, 'static/joint/data/')
        target4 = os.path.join(APP_ROOT, 'static/joint/')
        target5 = os.path.join(APP_ROOT, 'static/joint/line_chart/')
        target6 = os.path.join(APP_ROOT, 'static/joint/scatter_chart/')
        if not os.path.isdir(target):
            os.mkdir(target)
        if not os.path.isdir(target4):
            os.mkdir(target4)
        if not os.path.isdir(target1):
            os.mkdir(target1)
        if not os.path.isdir(target2):
            os.mkdir(target2)
        if not os.path.isdir(target3):
            os.mkdir(target3)
        if not os.path.isdir(target5):
            os.mkdir(target5)
        if not os.path.isdir(target6):
            os.mkdir(target6)
            
        filename = f.filename
        destination = "/".join([target1, f.filename])
        f.save(destination)
        original = target1
        processed = target2
        data = target3
        line_chart = target5
        scatter_chart = target6
        
        original_video = "/".join([original, filename])
        fname = str(filename.split(".")[:-1]).replace("['","").replace("']","")
        processed_video = "/".join([processed, fname+".mp4"])
        excel_data = "/".join([data, fname+".csv"])
        target5dir = os.path.join(APP_ROOT, 'static/joint/line_chart/%s/'%(fname))
        target6dir = os.path.join(APP_ROOT, 'static/joint/scatter_chart/%s/'%(fname))
        
        if not os.path.isdir(target5dir):
            os.mkdir(target5dir)
        if not os.path.isdir(target6dir):
            os.mkdir(target6dir)
        
        if not os.path.isfile(excel_data or processed):
            columnLst = ["frameNo"]
            for i in range(1, 18):
                xs = "x" + str(i)
                ys = "y" + str(i)
                columnLst = columnLst + [xs, ys]
            cap = cv2.VideoCapture(original_video)
            total_frames = int(cap.get(7))
            cap.release()
            
            
            vid = imageio.get_reader(original_video,  'ffmpeg')
            fps = vid.get_meta_data()['fps']
            writer = imageio.get_writer(processed_video, fps=fps)
            file_Lst = []
        else:
            print('已經有檔案了')
            options = 10
            total_frames = 1
            
            
    # elif(options == '2'):
    #     print('臉64')
    #     target = os.path.join(APP_ROOT, 'static/')
    #     target1 = os.path.join(APP_ROOT, 'static/face/original/')
    #     target2 = os.path.join(APP_ROOT, 'static/face/processed/')
    #     target3 = os.path.join(APP_ROOT, 'static/face/data/')
    #     target4 = os.path.join(APP_ROOT, 'static/face/')
    #     if not os.path.isdir(target):
    #         os.mkdir(target)
    #     if not os.path.isdir(target4):
    #         os.mkdir(target4)
    #     if not os.path.isdir(target1):
    #         os.mkdir(target1)
    #     if not os.path.isdir(target2):
    #         os.mkdir(target2)
    #     if not os.path.isdir(target3):
    #         os.mkdir(target3)
        
    #     filename = f.filename
    #     destination = "/".join([target1, f.filename])
    #     f.save(destination)
    #     original = target1
    #     processed = target2
    #     data = target3
        
    #     original_video = "/".join([original, filename])
    #     fname = str(filename.split(".")[:-1]).replace("['","").replace("']","")
    #     processed_video = "/".join([processed, fname+".mp4"])
    #     excel_data = "/".join([data, fname+".csv"])
        
    #     if not os.path.isfile(excel_data or processed):
    #         columnLst = ["frameNo"]
    #         for i in range(1, 69):
    #             xs = "x" + str(i)
    #             ys = "y" + str(i)
    #             columnLst = columnLst + [xs, ys]
    #         cap = cv2.VideoCapture(original_video)
    #         total_frames = int(cap.get(7))
    #         cap.release()
        
    #         vid = imageio.get_reader(original_video,  'ffmpeg')
    #         fps = vid.get_meta_data()['fps']
    #         writer = imageio.get_writer(processed_video, fps=fps)
    #         file_Lst = []
    #     else:
    #         print('已經有檔案了')
    #         options = 10
    #         total_frames = 1
            
            
        # elif(options == '3'):
        #     print('臉82')
        #     target = os.path.join(APP_ROOT, 'static/')
        #     target1 = os.path.join(APP_ROOT, 'static/dlib_face/original/')
        #     target2 = os.path.join(APP_ROOT, 'static/dlib_face/processed/')
        #     target3 = os.path.join(APP_ROOT, 'static/dlib_face/data/')
        #     target4 = os.path.join(APP_ROOT, 'static/dlib_face/')
        #     if not os.path.isdir(target):
        #         os.mkdir(target)
        #     if not os.path.isdir(target4):
        #         os.mkdir(target4)
        #     if not os.path.isdir(target1):
        #         os.mkdir(target1)
        #     if not os.path.isdir(target2):
        #         os.mkdir(target2)
        #     if not os.path.isdir(target3):
        #         os.mkdir(target3)
                
                
                
    
        
        
        # 这里要产生一个任务token 接下来才能查询进度
        # 返回临时用token A123456
        #token = "A123456"
        # 产生Token 用uuid
    token = str(uuid.uuid1())
    data = {"token":token}
    #记录使用者的token
    session[token] = 0
    #回传数据
    return jsonify(data), 200
        

        
        



@app.route('/getprocess', methods=['POST'])
def getProcess():
    #从token判断使用者
    #print('选择的选项是: '+ options)
    token = request.form.get('token')
    #print('recieve :' + token)
    frame_count = session[token]
    start = time.time()
    if(options == '1'):
        print('關節點處理')
        frame = vid.get_data(frame_count)  # Capture frame-by-frame
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
        if(len(kp) != 0):
            frame_Lst = draw_all_skeletons(frame_count,img0, kp)
        else:
            frame_Lst = [frame_count]
            for i in range(17):
                x = 0
                y = 0
                frame_Lst = frame_Lst + [x, y]
            file_Lst.append(frame_Lst)
        print(frame_Lst)
        file_Lst.append(frame_Lst)
        img1 = np.append(frame,img0, axis=1)#把 2 張 img 接起來
        writer.append_data(img1)
            
            
    # elif(options == '2'):
    #     print('臉64處理')
    #     frame = vid.get_data(frame_count)  # Capture frame-by-frame
    #     img0 = np.copy(frame)
    #     # start = time.time()
    #     preds = fa.get_landmarks(frame)
    #     # goal = time.time()
    #     # print('花費%s秒'%(goal - start))
    #     # print(preds)
    #     if(preds != None):
    #         frame_Lst = [frame_count]
    #         for faceIdx in range(1):
    #             for keyPoint in range(preds[0].shape[0]):
    #                 x = preds[faceIdx][keyPoint][0]
    #                 y = preds[faceIdx][keyPoint][1]
    #                 cv2.circle(img0, (x, y), radius=2, color=(255, 255, 255), thickness=3) 
    #                 frame_Lst = frame_Lst + [x, y]
    #         file_Lst.append(frame_Lst)
    #     else:
    #         frame_Lst = [frame_count]
    #         for i in range(68):
    #             x = 0
    #             y = 0
    #             frame_Lst = frame_Lst + [x, y]
    #         file_Lst.append(frame_Lst)
    #     file_Lst.append(frame_Lst)
    #     img1 = np.append(frame,img0, axis=1)#把 2 張 img 接起來
    #     writer.append_data(img1)
        
        
        
        
        
        
        
    #计算完成度
    
    frame_count = frame_count + 1
    session[token] =frame_count
    # 回传数据
    data = {'current':frame_count ,'total':total_frames}
    goal = time.time()
    print('花費%s秒'%(goal - start))
    print(data)
    return jsonify(data),200
        
    
    
        
    

@app.route("/show")
def index():
    if(options != 10):
        writer.close()
        df = pd.DataFrame(file_Lst, columns = columnLst)
        df.to_csv(excel_data, index = False)
    if(show_way == '1'):
        return render_template("show_videos.html", name2 = str("joint/processed/" + processed_video.split('/')[-1]))
    elif(show_way == '2'):
        columns,alldata = read_csv(excel_data)
        return render_template("show_data.html",name = str("joint/processed/" + excel_data.split('/')[-1]) ,len_columns = len(columns), columns = columns ,len_alldata1 = len(alldata),len_alldata2 = len(alldata[0]), alldata = alldata)
    elif(show_way == '3'):
        if len(os.listdir(target6dir)) == 0:
            generate_chart(excel_data,target5dir,target6dir)
        filelst = []
        for file in os.listdir(target6dir):
            filelst.append(os.path.join("./static/joint/scatter_chart/%s/"%(fname),file))
        return render_template("show_chart.html",file = filelst,lenfilelst = len(filelst))
    elif(show_way == '4'):
        if len(os.listdir(target5dir) ) == 0:
            generate_chart(excel_data,target5dir,target6dir)
        filelst = []
        for file in os.listdir(target5dir):
            filelst.append(os.path.join("./static/joint/line_chart/%s/"%(fname),file))
        return render_template("show_chart.html",file = filelst,lenfilelst = len(filelst))


if __name__ == '__main__':
    app.run(host="140.138.143.167", port=5000,threaded=True)



