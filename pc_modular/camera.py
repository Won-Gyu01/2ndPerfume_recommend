import cv2
from ultralytics import YOLO
import time

angry, fear, happy, neutral, sad, nan = 0, 0, 0, 0, 0, 0

def count_mood(x):
    global angry
    global fear
    global happy
    global neutral
    global sad
    global nan
    if(x==1):
        angry=angry+1
        return angry
    elif(x==2):
        fear=fear+1
        return fear
    elif(x==3):
        happy=happy+1
        return happy
    elif(x==4):
        neutral=neutral+1
        return neutral
    elif(x==5):
        sad=sad+1
        return sad
    else:
        nan=nan+1
        return nan

def print_results(res):
    print("======result======")
    print("nan :",res[0])
    print("angry :",res[1])
    print("fear :",res[2])
    print("happy :",res[3])
    print("neutral :",res[4])
    print("sad :",res[5])
    print("======result======")

def print_output(x):
    out = ['nan','anger', 'fear', 'happy', 'neutral', 'sad']
    if(x==0):
        print(out[0])
    elif(x==1):
        print(out[1])
    elif(x==2):
        print(out[2])
    elif(x==3):
        print(out[3])
    elif(x==4):
        print(out[4])
    elif(x==5):
        print(out[5])
    else:
        print('error')
    


def cameramood(modeldir):

    # Load the YOLOv8 model
    model = YOLO(modeldir)

    # Open the video file
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        start = time.time()


    # Loop through the video frames
    while cap.isOpened():
        now = time.time()
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            if results[0].boxes:
                clss= results[0].boxes.cls.cpu().detach().numpy().tolist()[0]
                clss = int(clss+1)
            else:
                clss=int(0)

            

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            #cv2.imshow("YOLOv8 Inference", annotated_frame)
            print(clss)
            count_mood(clss)
            if now-start >= 10:
                break

            # Break the loop if 'q' is pressed
        #if cv2.waitKey(1) & 0xFF == ord("q"):
            #    break
        else:
            # Break the loop if the end of the video is reached
            break

    #print result

    res = [nan,angry, fear, happy, neutral, sad]
    output = res.index(max(res))
    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    print_results(res)
    print("======output======")
    print("index :",output)
    print_output(output)
    print("======output======")
    return output



