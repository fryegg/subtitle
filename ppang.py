from pytube import YouTube
import moviepy.editor as mp
import os
import whisper
import cv2
import math

if not os.path.isfile('audio.mp3'):
    yt = YouTube('https://www.youtube.com/shorts/QfmYJW4Y0C4')

    yt.streams.filter(file_extension='mp4').get_by_resolution('720p').download(output_path='.', filename='input.mp4')


    clip = mp.VideoFileClip("input.mp4")
    clip.audio.write_audiofile("audio.mp3")

model = whisper.load_model("medium")
result = model.transcribe("audio.mp3", task='translate')
for r in result['segments']:
    print(f'[{r["start"]} --> {r["end"]}] {r["text"]}')


cap = cv2.VideoCapture('input.mp4')

w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (w, h))

frame_num = 0

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    cur_sec = frame_num / fps

    result_img = img.copy()

    img.flags.writeable = False
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    text = ''
    for r in result['segments']:
        if r['start'] < cur_sec <= r['end']:
            text = r['text']

    # x1 = int(detection.location_data.relative_bounding_box.xmin * w)
    # y1 = int(detection.location_data.relative_bounding_box.ymin * h) - 50
    
    x1 = int(w/2)
    y1 = int(h*(5/6))
    for i in range(math.ceil(len(text)/40)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        font_coords = (x1, y1+(i*25))
        t = text[i*40:(i+1)*40].strip()
        textsize = cv2.getTextSize(text = t, fontFace=font, fontScale=font_scale, thickness = font_thickness)[0]
        update_coords =  (font_coords[0] - int(textsize[0]/2), font_coords[1] - int(textsize[1]/2))

        cv2.putText(result_img, text=t, org=update_coords, fontFace=font, fontScale=font_scale, color=(0, 0, 0), thickness=font_thickness*5)
        cv2.putText(result_img, text=t, org=update_coords, fontFace=font, fontScale=font_scale, color=(255, 255, 255), thickness=font_thickness)

    out.write(result_img)

    frame_num += 1

cap.release()
out.release()