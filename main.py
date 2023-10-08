from sshtunnel import SSHTunnelForwarder
import cv2
import numpy as np
import time
import configparser
import subprocess

config = configparser.ConfigParser()
config.read('config.cfg')

ssh_bridge = config.getboolean('settings', 'ssh_bridge')
ssh_server_address = config.get('settings', 'ssh_server_address')
ssh_server_port = config.getint('settings', 'ssh_server_port')
ssh_username = config.get('settings', 'ssh_username')
ssh_password = config.get('settings', 'ssh_password')
remote_bind_address = config.get('settings', 'remote_bind_address')
remote_bind_port = config.getint('settings', 'remote_bind_port')
local_bind_address = config.get('settings', 'local_bind_address')
local_bind_port = config.getint('settings', 'local_bind_port')
wait_time = config.getint('settings', 'wait_time')
seek_person = config.getboolean('settings', 'seek_person')
refresh_detection_time = config.getint('settings', 'refresh_detection_time', fallback=5)
welcome_time = config.getint('settings', 'welcome_time', fallback=0)
remote_http_auth = config.getboolean('settings', 'remote_http_auth')
remote_http_username = config.get('settings', 'remote_http_username')
remote_http_password = config.get('settings', 'remote_http_password')

use_rtsp = config.getboolean('settings', 'use_rtsp', fallback=False)
rtsp_camera_ip = config.get('settings', 'rtsp_camera_ip')
rtsp_camera_username = config.get('settings', 'rtsp_camera_username')
rtsp_camera_password = config.get('settings', 'rtsp_camera_password')

N = config.getint('settings', 'heavy_model_frame_skip', fallback=15)

x_f = 0
y_f = 0
w_f = 0
h_f = 0

SIZE_X = 640
SIZE_Y = 480

previous_frame = None

def run_shell_script(script_name, background=True):
    if background:
        subprocess.Popen(['sh', script_name])
    else:
        process = subprocess.Popen(['sh', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout.decode())
        if stderr:
            print(f"Error executing {script_name}:")
            print(stderr.decode())

def has_significant_change(current_frame, threshold=0.001):
    global previous_frame
    
    current_frame = cv2.GaussianBlur(current_frame, (5, 5), 0)
    
    if previous_frame is None:
        previous_frame = current_frame
        return True
    
    diff = cv2.absdiff(current_frame, previous_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    _, thresholded_diff = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3),np.uint8)
    thresholded_diff = cv2.erode(thresholded_diff, kernel, iterations = 1)
    thresholded_diff = cv2.dilate(thresholded_diff, kernel, iterations = 1)
    
    changed_ratio = cv2.countNonZero(thresholded_diff) / (current_frame.shape[0] * current_frame.shape[1])
    previous_frame = current_frame
    has_significant_changes = changed_ratio > threshold
    return has_significant_changes



def videoStream(url):
    print("Starting video stream from {}...".format(url))
    if use_rtsp:
        rtsp_url = f"rtsp://{rtsp_camera_username}:{rtsp_camera_password}@{rtsp_camera_ip}:554/stream1"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    elif remote_http_auth:
        url_with_auth = "http://{}:{}@".format(remote_http_username, remote_http_password) + url.lstrip("http://")
        cap = cv2.VideoCapture(url_with_auth)
    else:
        cap = cv2.VideoCapture('{}'.format(url))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')  # model Haar cascade
    body_cascade = cv2.CascadeClassifier('models/haarcascade_fullbody.xml')  # model Haar cascade
    lbp_cascade = cv2.CascadeClassifier('models/lbpcascade_frontalface_improved.xml')  # model LBP cascade
    lbp_profile_cascade = cv2.CascadeClassifier('models/lbpcascade_profileface.xml')  # model LBP cascade
    net = cv2.dnn.readNetFromDarknet('models/yolov3.cfg', 'models/yolov3.weights') # model YOLOv3s
    found_text = "No person detected"
    frame_count = 0
    person_found = False
    last_detected_time = None
    person_detected_time = None
    welcome_detection_time = time.time()
    layer_names = net.getUnconnectedOutLayersNames()

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to fetch frame.")
            break
        
        frame_count += 1

        #frame = cv2.resize(frame, (SIZE_X, SIZE_Y))
        
        #print("Frame size: {}x{}".format(frame.shape[1], frame.shape[0]))
        #print("person_found: {}".format(person_found))
        #print("last_detected_time: {}".format(last_detected_time))
        #print("refresh_detection_time: {}".format(refresh_detection_time))
        if not person_found or not last_detected_time or time.time() - last_detected_time > refresh_detection_time:      
            
            if has_significant_change(frame):
                #print("significant change detected")
                if(seek_person):
                    #print("searching for person...")
                    found_text = "No person detected"
                    person_found = False
                    x_f = 0
                    y_f = 0
                    w_f = 0
                    h_f = 0
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                        
                    faces = lbp_profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
                    for (x, y, w, h) in faces:
                        x_f = x
                        y_f = y
                        w_f = w
                        h_f = h
                        found_text = "Face1 detected"
                        last_detected_time = time.time()
                        person_found = True
                        break  

                    if not person_found:
                        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

                        for (x, y, w, h) in faces:
                            x_f = x
                            y_f = y
                            w_f = w
                            h_f = h
                            found_text = "Face2 detected"
                            last_detected_time = time.time()
                            person_found = True
                            break

                    if not person_found:
                        bodies = lbp_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
                        for (x, y, w, h) in bodies:
                            x_f = x
                            y_f = y
                            w_f = w
                            h_f = h
                            found_text = "Face3 detected"
                            last_detected_time = time.time()
                            person_found = True
                            break

                    if not person_found:
                        bodies = body_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)
                        for (x, y, w, h) in bodies:
                            x_f = x
                            y_f = y
                            w_f = w
                            h_f = h
                            found_text = "Body detected"
                            last_detected_time = time.time()
                            person_found = True
                            break

                    if not person_found:
                        
                        if frame_count % N == 0:
                            
                            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

                            net.setInput(blob)
                            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                            net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
                            
                            detections = net.forward(layer_names)
                            
                            for detection in detections:
                                for obj in detection:
                                    scores = obj[5:]
                                    class_id = np.argmax(scores)
                                    confidence = scores[class_id]
                                    if confidence > 0.5:
                                        if class_id == 0:
                                            person_found = True
                                            last_detected_time = time.time()
                                            break
                    if person_detected_time is None:
                        person_detected_time = last_detected_time
            #else:
            #    print("no significant change detected")
                
        if not person_found or found_text == "No person detected":
            cv2.putText(frame, "No person detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x_f, y_f), (x_f+w_f, y_f+h_f), (255, 0, 0), 2)
            if person_detected_time is not None and time.time() - person_detected_time > refresh_detection_time:
                person_detected_time = time.time()
                run_shell_script("./routines/detected.sh")
                if welcome_time > 0 and time.time() - welcome_time > welcome_detection_time:
                    welcome_detection_time = time.time()
                    run_shell_script("./routines/welcome.sh")
            cv2.putText(frame, found_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Frame', frame)

        # push 'q' to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

def main():
    if (ssh_bridge):
        with SSHTunnelForwarder(
            (ssh_server_address, ssh_server_port),
            ssh_username=ssh_username,
            ssh_password=ssh_password,
            remote_bind_address=(remote_bind_address, remote_bind_port),
            local_bind_address=(local_bind_address, local_bind_port)
        ) as tunnel:
            print(f'Tunnel established on localhost:{tunnel.local_bind_port}')

            url = 'http://localhost:{}/'.format(tunnel.local_bind_port)
            if wait_time > 0:
                print('Waiting {} seconds for manual checks before start...'.format(wait_time))
                time.sleep(wait_time)

            videoStream(url)
    else:
        url = 'http://{}:{}/'.format(remote_bind_address, remote_bind_port)
        if wait_time > 0:
            print('Waiting {} seconds for manual checks before start...'.format(wait_time))
            time.sleep(wait_time)

        videoStream(url)

if __name__ == '__main__':
    main()

