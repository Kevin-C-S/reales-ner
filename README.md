# Welcome to ner-reales!

This library was developed by **Los Reales** team with the purpose of helping the identification and classification of news related to environmental impact in the Amazon.
Developed at **CodeFest Ad Astra**


# 💡 Prerequisites

**python3**


## How to Run
YOU NEED TO RUN IN YOUR CONSOLE (BETTER IF DONE IN A VENV) THE COMMAND:
********
On linux:
```
apt-get install libgl1-mesa-glx -y
pip install opencv-python
apt install tesseract-ocr -y
apt install libtesseract-dev -y
```

Then run:
```
pip install -r requirements.txt
pip install pip install imbalanced-learn
wget -P reales_ner/ https://www.dropbox.com/s/1imxkvesu7iy4i0/yolo_custom_model.pt
python -m spacy download es_core_news_sm
```


********

### For the text challenge:

Then import the module "reales_ner.ner" in the file you want to use the functions.
Great, now you can get the jason with the entities and classification by text, file and url. The methods are:
ner_from_str(text, output_path) 
ner_from_file(text_path, output_path)
ner_from_url(url, output_path)
********

### For the object detection challenge:
Then import the module "reales_ner.impro" in the file you want to use the functions.

You only should call the function detect_objects_in_video(video_path, output_path).


Other methods are auxiliar and should not be called.

Aditionally the file test.py has an example of how it should be excecuted.

## 🐸  Aloha!
![Team los Reales](https://firebasestorage.googleapis.com/v0/b/moviles2023-c0911.appspot.com/o/images%2Fguardado.PNG?alt=media&token=ad60f1da-1dc1-433d-bcf3-af355bb1857f&_gl=1*121x8cv*_ga*NTI4Mjc5OTcyLjE2Nzk3MDEzMjM.*_ga_CW55HF8NVT*MTY4NTgwNDY2NC4xNi4xLjE2ODU4MDQ4MjkuMC4wLjA.)


