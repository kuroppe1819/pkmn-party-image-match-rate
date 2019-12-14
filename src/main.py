import cv2
SAMPLE_DEFAULT_IMAGE = "./image_model/pkmn_rental_party_5.jpeg"
OUTPUT_IMAGE_WIDTH = 480

def resizeImageKeepAspectRatio(width, img):
    height = int(width / img.shape[1] * img.shape[0])
    return cv2.resize(img, (width, height))

pkmnImg = cv2.imread(SAMPLE_DEFAULT_IMAGE)
resizeImg = resizeImageKeepAspectRatio(width = OUTPUT_IMAGE_WIDTH, img = pkmnImg)

# img[top : bottom, left : right]
resizeImgHeight = resizeImg.shape[0]
footerTopY = int(resizeImgHeight * 4 / 5)
croppedPkmnPartyImg = resizeImg[0 : footerTopY, 0 : OUTPUT_IMAGE_WIDTH]
croppedTrainerIdImg = resizeImg[footerTopY : resizeImgHeight, 0 : OUTPUT_IMAGE_WIDTH]
cv2.imwrite("./output/cropped_pkmn_party.jpeg", croppedPkmnPartyImg)
cv2.imwrite("./output/cropped_trainer_id_.jpeg", croppedTrainerIdImg)

# グレースケール変換
croppedPkmnPartyGrayImg = cv2.cvtColor(croppedPkmnPartyImg, cv2.COLOR_BGR2GRAY)
cannyImg = cv2.Canny(croppedPkmnPartyGrayImg, 50, 100)
# cv2.namedWindow("croppedPkmnPartyGrayImg")
# cv2.imshow("croppedPkmnPartyGrayImg", croppedPkmnPartyGrayImg)
cv2.namedWindow("cannyImg")
cv2.imshow("cannyImg", cannyImg)

cv2.waitKey(0)
cv2.destroyAllWindows()