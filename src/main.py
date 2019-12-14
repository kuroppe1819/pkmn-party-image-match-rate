import cv2
SAMPLE_DEFAULT_IMAGE = "./image_model/pkmn_rental_party_5.jpeg"
OUTPUT_IMAGE_WIDTH = 480

def resizeImageKeepAspectRatio(width, img):
    height = int(width / img.shape[1] * img.shape[0])
    return cv2.resize(img, (width, height))

def decreaseColor(color):
    if color < 64:
        return 32
    elif color < 128:
        return 96
    elif color < 196:
        return 160
    else:
        return 224

def decreaseRgbOfImg(img):
    for i, row in enumerate(img):
        for k, pxValue in enumerate(row):
            for m in range(3): #RGBの配列を参照する
                img[i, k][m] = decreaseColor(pxValue[m])

pkmnImg = cv2.imread(SAMPLE_DEFAULT_IMAGE)
resizeImg = resizeImageKeepAspectRatio(width = OUTPUT_IMAGE_WIDTH, img = pkmnImg)
decreaseRgbOfImg(resizeImg)

# img[top : bottom, left : right]
resizeImgHeight = resizeImg.shape[0]
footerTopY = int(resizeImgHeight * 4 / 5)
croppedPkmnPartyImg = resizeImg[0 : footerTopY, 0 : OUTPUT_IMAGE_WIDTH]
croppedTrainerIdImg = resizeImg[footerTopY : resizeImgHeight, 0 : OUTPUT_IMAGE_WIDTH]
cv2.imwrite("./output/cropped_pkmn_party.jpeg", croppedPkmnPartyImg)
cv2.imwrite("./output/cropped_trainer_id.jpeg", croppedTrainerIdImg)

# グレースケール変換
croppedPkmnPartyGrayImg = cv2.cvtColor(croppedPkmnPartyImg, cv2.COLOR_BGR2GRAY)
cannyImg = cv2.Canny(croppedPkmnPartyGrayImg, 50, 100)
# cv2.namedWindow("croppedPkmnPartyGrayImg")
# cv2.imshow("croppedPkmnPartyGrayImg", croppedPkmnPartyGrayImg)
cv2.namedWindow("cannyImg")
cv2.imshow("cannyImg", cannyImg)

cv2.waitKey(0)
cv2.destroyAllWindows()