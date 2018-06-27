from threading import Thread
from multiprocessing import Queue
import os
import cv2
import time
import qiniu
from ava_auth import AuthFactory
import requests
import json
from argparse import ArgumentParser

stop_signal = False
frame_data_queue = Queue()
frame_url_queue = Queue()

###########functions about get frame and add frame to frame data queue###############

def set_cap():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    time.sleep(2)
    return cap
    
def get_frames(cap):
    global stop_signal
    while not stop_signal:
        _, frame = cap.read()
        framename = str("%.07f" %time.time())
        frame_data_queue.put((framename,frame))
        cv2.imshow("capture", frame)
        if cv2.waitKey(50) & 0xff == ord("q"):
            stop_signal = True

############functions about save frame and upload them to bucket and add url to frame_url_queue#######

upload_access_key = "********************************"
upload_secret_key = "********************************"
bucket_name = "framedecpose"
bucket_url = "your bucket url like http://pargr4az5.bkt.clouddn.com/"
upload_auth = qiniu.Auth(upload_access_key, upload_secret_key)

def save_frame(frame_data, dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

    filename = frame_data[0] + ".jpg"
    filepath = os.path.join(dir, filename)
    frame = frame_data[1]
    cv2.imwrite(filepath, frame)
    filelistpath = os.path.join(dir, "frame_list.txt")
    with open(filelistpath, "a") as f:
        f.write(filename + "\n")

    return filepath

def upload_single_frame(filepath):
    filename = os.path.basename(filepath)
    upload_token = upload_auth.upload_token(bucket_name, filename, 3600)
    ret, _ = qiniu.put_file(upload_token, filename, filepath)
    return ret["hash"] == qiniu.etag(filepath)
    
def upload_frame(filepath, reupload=3):
    upload_success = False
    while not upload_success and reupload:
        upload_success = upload_single_frame(filepath)
        reupload -= 1
    if upload_success:
        frame_url = bucket_url + os.path.basename(filepath)
        frame_url_queue.put(frame_url)
        print("upload->  " + frame_url + "  sucess!")
    else:
        print("upload->  " + frame_url + "  failed!")
    return upload_success

def save_frames(frames_dir):
    while not stop_signal or not frame_data_queue.empty():
        if not frame_data_queue.empty():
            frame_data = frame_data_queue.get()
            filepath = save_frame(frame_data, frames_dir)
            upload_frame(filepath)


###############functions about process frames

process_access_key = "******************************"
process_secret_key = "******************************"
header = {"Content-Type":"application/json"}

detect_url = "http://serve.atlab.ai/v1/eval/facex-detect"
pose_url = "http://serve.atlab.ai/v1/eval/facex-pose"
process_auth = AuthFactory(process_access_key, process_secret_key).get_qiniu_auth()

def detect_frame(fileurl):
    data = {"data":{"uri":fileurl}}
    r = requests.post(detect_url, None, data, headers=header, auth=process_auth)
    contentObj = json.loads(r.content)
    # print("detected->  " + fileurl)
    return contentObj["result"]
    
def pose_frame(fileurl, det_rst):
    if len(det_rst["detections"]) != 0:
        data = {"data":{"uri":fileurl, "attribute":det_rst}}
        r = requests.post(pose_url, None, data, headers=header, auth=process_auth)
        contentObj = json.loads(r.content)
        # print("posed->  " + fileurl)
        # print(json.dumps(data))
        # print(r.content)
        if "error" in contentObj.keys():
            return {"landmarks":[]}
        else:
            return contentObj["result"]
    else:
        return {"landmarks":[]}

def save_result(rstdir, fileurl, det_rst, pose_rst):
    if not os.path.exists(rstdir):
        os.mkdir(rstdir)
    with open(os.path.join(rstdir, "detect_result.json"), "a") as f:
        result = {"url":fileurl, "result":det_rst}
        f.write(json.dumps(result) + "\n")
    with open(os.path.join(rstdir, "pose_result.json"), "a") as f:
        result = {"url":fileurl, "result":pose_rst}
        f.write(json.dumps(result) + "\n")

def mark_frame_from_frame(frame, filename, rstdir, det_rst, pose_rst):
    if not os.path.exists(rstdir):
        os.mkdir(rstdir)
    for detection in det_rst["detections"]:
        if detection["class"] == "face":
            topleft = (int(detection["pts"][0][0]), int(detection["pts"][0][1]))
            bottomright = (int(detection["pts"][2][0]), int(detection["pts"][2][1]))
            cv2.rectangle(frame, topleft, bottomright, (255, 255, 0), 2)
    for landmark in pose_rst["landmarks"]:
        topleftpoint = [99999, 99999]
        for point in landmark["landmark"]:
            point = (int(point[0]), int(point[1]))
            cv2.circle(frame, point, 1, (0, 255, 255), 1)
            if point[0] < topleftpoint[0]:
                topleftpoint[0] = point[0]
            if point[1] < topleftpoint[1]:
                topleftpoint[1] = point[1]
        # print(topleftpoint)
        text = str(int(landmark["pos"][0])) + ", " + str(int(landmark["pos"][1])) + ", " + str(int(landmark["pos"][2]))
        cv2.putText(frame, text, (topleftpoint[0], topleftpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
    cv2.imwrite(os.path.join(rstdir, os.path.basename(filename)), frame)

def mark_frame_from_file(srcfilepath, rstdir, det_rst, pose_rst):
    if not os.path.exists(rstdir):
        os.mkdir(rstdir)
    frame = cv2.imread(srcfilepath)
    for detection in det_rst["detections"]:
        if detection["class"] == "face":
            topleft = (int(detection["pts"][0][0]), int(detection["pts"][0][1]))
            bottomright = (int(detection["pts"][2][0]), int(detection["pts"][2][1]))
            cv2.rectangle(frame, topleft, bottomright, (255, 255, 0), 2)
    for landmark in pose_rst["landmarks"]:
        topleftpoint = [99999, 99999]
        for point in landmark["landmark"]:
            point = (int(point[0]), int(point[1]))
            cv2.circle(frame, point, 1, (0, 255, 255), 1)
            if point[0] < topleftpoint[0]:
                topleftpoint[0] = point[0]
            if point[1] < topleftpoint[1]:
                topleftpoint[1] = point[1]
        text = str(int(landmark["pos"][0])) + ", " + str(int(landmark["pos"][1])) + ", " + str(int(landmark["pos"][2]))
        cv2.putText(frame, text, (topleftpoint[0], topleftpoint[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(rstdir, os.path.basename(srcfilepath)), frame)
    print("marked->  " + srcfilepath)

def process_frames(srcdir, rstdir):
    while not stop_signal or not frame_url_queue.empty():
        if not frame_url_queue.empty():
            frameurl = frame_url_queue.get()
            det_rst = detect_frame(frameurl)
            pose_rst = pose_frame(frameurl, det_rst)
            save_result(rstdir, frameurl, det_rst, pose_rst)
            srcfilepath = os.path.join(srcdir, os.path.basename(frameurl))
            mark_frame_from_file(srcfilepath, rstdir, det_rst, pose_rst)

#######################combine the frames into video

def combine_frames_into_video(srcdir, rstdir):
    framefiles = []
    with open(os.path.join(srcdir, "frame_list.txt")) as f:
        framefiles = [line.strip() for line in f.readlines()]
    frames_time = float(os.path.splitext(framefiles[-1])[0]) - float(os.path.splitext(framefiles[0])[0])
    fps = int(len(framefiles)/frames_time)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')  
    src_video_writer = cv2.VideoWriter(os.path.join(srcdir, "src_video.mp4"), fourcc, fps, (1280, 720))
    rst_video_writer = cv2.VideoWriter(os.path.join(rstdir, "rst_video.mp4"), fourcc, fps, (1280, 720))
    for framefile in framefiles:
        srcframe = cv2.imread(os.path.join(srcdir, framefile))
        rstframe = cv2.imread(os.path.join(rstdir, framefile))
        # print(srcframe)
        src_video_writer.write(srcframe)
        rst_video_writer.write(rstframe)
    src_video_writer.release()
    rst_video_writer.release()



if __name__ == "__main__":
    ap = ArgumentParser('draw boxs and points')
    ap.add_argument('-d', '--dir', required=True, type=str,
                    help='dir to save frames data')
    args = ap.parse_args()
    dir = args.dir

    if os.path.exists(dir):
        print("dir already exists!")
        os._exit(0)
    else:
        os.mkdir(dir)
    srcdir = os.path.join(dir, "src_dir")
    rstdir = os.path.join(dir, "rst_dir")
    # get_frame_thread = Thread(target = get_frames, args=(set_cap(),))
    save_frame_thread = Thread(target = save_frames, args=(srcdir,))
    process_frame_thread = Thread(target = process_frames, args=(srcdir, rstdir))
    
    # get_frame_thread.start()
    save_frame_thread.start()
    process_frame_thread.start()

    get_frames(set_cap())

    while save_frame_thread.is_alive() or process_frame_thread.is_alive():
        # print("There are already " + str(frame_data_queue.qsize()) + " frames last to upload!")
        # print("There are already " + str(frame_url_queue.qsize()+frame_data_queue.qsize()) + " frames last to process!")
        time.sleep(5)
    print("Combining frames into video...")
    combine_frames_into_video(srcdir, rstdir)
    print("All is done!")


