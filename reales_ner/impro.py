import pytesseract
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
from moviepy.editor import VideoFileClip


from imageai.Detection.Custom import CustomObjectDetection


def select_coordinate_string(strings):
    def is_coordinate_format(string):
        # Checks if the string matches the coordinate format
        import re
        pattern = r'\d{1,2}°\d{1,2}\'\d{1,2}\.\d{1,2}" [NSEW]'
        return re.match(pattern, string) is not None

    def evaluate_coordinate_string(string):
        # Evaluates the string based on coordinate-like characteristics
        score = 0
        if is_coordinate_format(string):
            score += 2
        if "N" in string or "W" in string:
            score += 1
        return score

    scores = [evaluate_coordinate_string(string) for string in strings]
    max_score = max(scores)

    if scores.count(max_score) == 1:
        index = scores.index(max_score)
        return strings[index]
    else:
        return None
    
    
def select_by_numbers(strings):
    def extract_number(string):
        # Extracts the maximum number from a string using regex
        import re
        numbers = re.findall(r'\d+', string)
        return max(numbers, key=int) if numbers else None

    max_number = max(map(extract_number, strings), key=lambda x: int(x) if x else float('-inf'))

    if max_number:
        for string in strings:
            if max_number in string:
                return string

    return None


def get_candidate(strings):
    possible_answer =select_coordinate_string(strings)
    if (possible_answer is None):
        p2_answer = select_by_numbers(strings)
        if (p2_answer is None):
            return ""
        else:
            return p2_answer

    else:
        return possible_answer


def add_letter_to_string(string, letter):
    try:
        if string[-1] != letter:
            string += letter
    except:
        pass
    return string




def perform_ocr(image,language='eng'):   
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply color-based preprocessing
    # Define the color thresholds for the text and background
    lower_text_color = np.array([0, 100, 0], dtype=np.uint8)  # Lower threshold for text color (light green)
    upper_text_color = np.array([110, 255, 110], dtype=np.uint8)  # Upper threshold for text color (light green)
    lower_bg_color = np.array([0, 0, 0], dtype=np.uint8)  # Lower threshold for background color (dark gray)
    upper_bg_color = np.array([255, 255, 255], dtype=np.uint8)  # Upper threshold for background color (dark gray)

    # Create binary masks for text and background colors
    text_mask = cv2.inRange(image, lower_text_color, upper_text_color)
    bg_mask = cv2.inRange(image, lower_bg_color, upper_bg_color)

    # Combine the masks to get the final binary mask
    final_mask = cv2.bitwise_and(text_mask, bg_mask)

    # Apply the binary mask to the grayscale image
    enhanced_gray = cv2.bitwise_and(gray, gray, mask=final_mask)

    # Perform OCR using Tesseract
    # Perform OCR using Tesseract
    c1 = pytesseract.image_to_string(image, lang=language).replace("\n", "").strip()
    c2 = pytesseract.image_to_string(gray, lang=language).replace("\n", "").strip()
    c3 = pytesseract.image_to_string(enhanced_gray, lang=language).replace("\n", "").strip()

    strings = [c1,c2,c3]
    text = get_candidate(strings)
    try:
        strings.remove(text) 
    except:
        pass
    text2 =get_candidate(strings)
    if text2.strip() == "":
        text2=text
    return text, text2

def process_image_left_left(image, roi=(2, 38, 62, 24), language='eng'):
    if roi is not None:
        x, y, w, h = roi
        image_roi = image[y:y+h, x:x+w]
    else:
        image_roi = image
        
    height, width = image_roi.shape[:2]

    new_image = np.full((height, width+50, 3), (64,64,64), dtype=np.uint8)
    color = (160, 160, 160)
    new_image[:, 25:width+25] = image_roi
    new_image[:, :25] = np.full((height, 25, 3), color, dtype=np.uint8)
    new_image[:, width+25:width+50] = np.full((height, 25, 3), color, dtype=np.uint8)

    # If ROI is specified, extract the region of interest from the image
    denoised_image = cv2.fastNlMeansDenoisingColored(new_image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    denoised_image0 = cv2.fastNlMeansDenoisingColored(denoised_image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    return perform_ocr(denoised_image0, language)
    
def process_image_left(image,language='eng'):
    text_left_left, text_left_left1 =process_image_left_left(image)
    text_left, text_left1 =process_image_left_left(image, (70, 38, 90, 25))
    return (text_left_left+" "+text_left, text_left_left1+" "+text_left1)

# Paramas:
# image: It is a cv2 frame but it could be an image_path if uncommented the third line
def process_image_right(image, roi=(154, 37, 250, 25), language='eng'):
     # If ROI is specified, extract the region of interest from the image
    if roi is not None:
        x, y, w, h = roi
        image_roi = image[y:y+h, x:x+w]
    else:
        image_roi = image
    # Display the cropped image
   # Apply a different color transformation to the first 20 pixels in X
    denoised_image = cv2.fastNlMeansDenoisingColored(image_roi, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    denoised_image0 = cv2.fastNlMeansDenoisingColored(denoised_image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return perform_ocr(denoised_image0)

def get_coordinates(image):
    right_text, right_text1  = process_image_right(image)
    left_text, left_text1 = process_image_left(image)
    option1 =add_letter_to_string(left_text, "N") + ";"+add_letter_to_string(right_text, "W") 
    option2 =add_letter_to_string(left_text1, "N") + ";"+add_letter_to_string(right_text1, "W") 
    return(option1, option2)

def getTime(image):
    spare =(0, 152, 140, 28)
    x, y, w, h = spare

    image_roi = image[y:y+h, x:x+w]
    
    left_text, left_text1 = process_image_left_left(image, (0,152, 60, 25))
    right_text, right_text1 = process_image_right(image, (58,152, 100, 25))
    return (left_text + right_text, left_text1+right_text1 )


detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('./reales_ner/yolo_custom_model.pt')
detector.setJsonPath("./reales_ner/data_reto_1_yolov3_detection_config.json")
detector.loadModel()

def convert_seconds_to_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_frame_info(frame,count,path):
    s = ""
    detections = detector.detectObjectsFromImage(input_image=frame,output_image_path=f'{path}/{count}-detected.jpg')
    if detections:
        coords = get_coordinates(frame)
        coords = '|'.join(list(coords)).replace(',','.')
        time = getTime(frame)
        time = '|'.join(list(time)).replace(',','.')
        for detection in detections:
            line=f'{count},{detection["name"]},{convert_seconds_to_time(count*5)},{time},{coords}\n'
            s+=line
    return s

def detect_objects_in_video(video_path, output_path):
    output_path = output_path.strip('/')
    f=open(output_path+'/output.csv', "w")
    SAVING_FRAMES_PER_SECOND = 0.2
    # load the video clip
    video_clip = VideoFileClip(video_path)

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(video_clip.fps, SAVING_FRAMES_PER_SECOND)
    # if SAVING_FRAMES_PER_SECOND is set to 0, step is 1/fps, else 1/SAVING_FRAMES_PER_SECOND
    step = 1 / video_clip.fps if saving_frames_per_second == 0 else 1 / saving_frames_per_second
    # iterate over each possible frame
    count = 0
    f.write('ID,OBJECT_TYPE,TIME_IN_RECORDING,TIME_RECORDED,COORDINATES_TEXT\n')
    for current_duration in np.arange(0, video_clip.duration, step):
        # save the frame with the current duration
        frame = video_clip.get_frame(current_duration)
        f.write(get_frame_info(frame,count,output_path))
        count += 1        
    f.close()