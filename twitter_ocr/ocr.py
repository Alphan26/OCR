import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
import cv2
import os
import time
import fastwer
# import numpy as np
# img = cv2.imread("pictures/paperback2.jpg")
# resized_img = cv2.resize(img, (1000,1000))
# #inverted_img = cv2.bitwise_not(img, cv2.INTER_LINEAR)
# grayscale_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
# # cv2.imshow("paperback2.jpg",resized_img)
# # cv2.waitKey(0)
# text_data = pytesseract.image_to_string(grayscale_img)
# # print(text_data)
accuracy_score = 100 - fastwer.score(list("mitten"),list("fitten"))
print(accuracy_score)


def read_img_data(filename):
    img = cv2.imread(filename)
    resized_img = cv2.resize(img, (1000,1000))
    #inverted_img = cv2.bitwise_not(img, cv2.INTER_LINEAR)
    grayscale_img = cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
    cv2.imshow(filename,resized_img)
    cv2.waitKey(0)
    #text_data = pytesseract.image_to_string(grayscale_img)
    #return text_data

def arrangeImages():
    # resize can be done
    # noise removal maybe
    # dilate , erode yapılabilir.
    # opencv tutoriala bi daha bakabilirsin.

# # txt ye yazmayı manuel değil de normal nasıl yapabilirim acaba
# def writeTextData():
#     for i in range(len(os.listdir("pictures/"))):
#         # resmi burada açacağım sonra yazıyı yazıp tekrar devam edeceğim.
#         text_data_str = "txtData" + str(i) + ".txt"
#         with open(text_data_str,"w") as f:
#             # resim bir yandan açıkken bir yandan yazı yazazbilir miyim ? 
#             f.write()


def cvtTxt2Str():
    txts = []
    for i in range(len(os.listdir("pictures/"))):
        text_data_str = "txtData" + str(i) + ".txt"
        with open(text_data_str,"r") as f:
            txts.append(str(f.read()))
    return txts

def compareTwoStrings():
    scores = []
    txt = []
    picture = []
    for i in range(len(os.listdir("pictures/"))):
        txt.append(cvtTxt2Str()[i])
        picture.append(read_img_data("pictures/" + os.listdir("pictures/")[i]))
        # print(len(os.listdir("pictures/")))
        scores.append(fastwer.score(list(cvtTxt2Str()[i]),list(read_img_data("pictures/" + os.listdir("pictures/")[i])),char_level = True))
        # print(scores)
    return scores


for i in range(len(os.listdir("pictures/"))):
    read_img_data("pictures/" + os.listdir("pictures/")[i])

#print(compareTwoStrings())
#print(os.listdir("pictures")[2])
# def noise_removal(image):
# kernel = np.ones((1,1), np.uint8)
# dilated_img = cv2.dilate(grayscale_img,kernel,iterations=1)
# kernel = np.ones((1,1), np.uint8)
# image = cv2.erode(image,kernel,iterations=1)
# image = cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
# image = cv2.medianBlur(image,3)
# return (image)

# dilated_img = noise_removal(grayscale_img)



