import os
from utils.box_plot import plot_bboxes
from ultralytics import YOLO
import cv2
from deepsort_tracker import Tracker
IN_COLAB = False


print(os.path)

if IN_COLAB:
    datasets_path = "/content"
    model_path = "drive/MyDrive/Models/runs/detect"
else:
    datasets_path = "../datasets"
    model_path = ""


model = YOLO(f'models/yolov8n_Person_Detection2/weights/best.pt')
tracker = Tracker()
cap = cv2.VideoCapture(f'{datasets_path}/2_man_walking.mp4')
video_frames = []
n = 0
tras = []
f = open("logs", "w", encoding="UTF-8")
while cap.isOpened():
    #try:
    ret, frame = cap.read()
    if frame is None:
        break
    results = model.predict(frame)

    detections = []
    detections1 = []
    for r in results[0].boxes.data:
        x1, y1, x2, y2, score, class_id = r
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        detections.append([x1, y1, x2, y2, float(score)])
        detections1.append([x1, y1, x2, y2, float(score), -1])
    if len(detections) == 0:
        cv2.imwrite(f'images/frame{n}.png', frame)
        cv2.imshow("resulted.jpeg", frame)
        n += 1
        continue
    tracker.update(frame, detections)
    track_ids = []
    tras.append(len(tracker.tracks))
    dets = []
    i = 0
    f.write(f"n_frame= {n} \n")
    f.write(f"detections= {detections}\n")
    for track in tracker.tracks:
        bbox = track.bbox
        x1, y1, x2, y2 = bbox
        print(i)
        if i == 1 and len(detections) < 2:
            f.write("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
        dets.append([*bbox, 1, track.track_id]) #detections[i][4]
        track_ids.append(track.track_id)
        i += 1


    f.write(f"dets= {dets}\n")
    f.write(f"track_ids= {track_ids}\n")
    im = plot_bboxes(frame, dets, score=True)
    im = plot_bboxes(im, detections1, score=True, colors=[(127, 127, 127)])
    cv2.imwrite(f'images/frame{n}.png', im)
    cv2.imshow("resulted.jpeg", im)
    n += 1
    if cv2.waitKey(1) == ord('q'):
        break
    #except Exception as e:
        #print(e)
cap.release()
cv2.destroyAllWindows()
