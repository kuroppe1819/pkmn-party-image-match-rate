import cv2
SAMPLE_DEFAULT_IMAGE = "./image_model/pkmn_rental_party_5.jpeg"
OUTPUT_IMAGE_WIDTH = 480

def resizeImageKeepAspectRatio(width, img):
    height = int(width / img.shape[1] * img.shape[0])
    return cv2.resize(img, (width, height))

pkmnImg = cv2.imread(SAMPLE_DEFAULT_IMAGE)
resizeImg = resizeImageKeepAspectRatio(width = OUTPUT_IMAGE_WIDTH, img = pkmnImg)
cv2.imwrite("./output/resize.jpeg", resizeImg)


