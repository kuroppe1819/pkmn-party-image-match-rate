import cv2
import sys
import glob
import numpy

OUTPUT_IMAGE_WIDTH = 480
OUTPUT_IMAGE_HEIGHT = 216
THRESHOLD_CONTOUR_AREA = 500
RECTANGLE_WIDTH = 104
RECTANGLE_HEIGHT = 68

def isShowRectangle(rect):
    return cv2.contourArea(rect) >= THRESHOLD_CONTOUR_AREA

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

def getAppropriatePkmnImg(numberOfParty):
    npy = numpy.load("./output/model/pkmn_party_imgs_model.npy")
    return npy[numberOfParty - 1]

def calcMatchRate(sourceImg, targetImg):
    pxCount = 0
    matchCount = 0
    for sourceRow, targetRow in zip(sourceImg, targetImg):
        for sourcePxValue, targetPxValue in zip(sourceRow, targetRow):
            pxCount += 1
            if (sourcePxValue == targetPxValue).all():
                matchCount += 1
    return matchCount / pxCount 

if __name__ == "__main__":
    argvLength = len(sys.argv)
    if argvLength == 1 and argvLength > 2:
        raise Exception("引数の数が不正です")

    imgPath = sys.argv[1]
    if len(glob.glob(imgPath)) > 1:
        raise Exception("引数が不正です")

    targetImg = cv2.imread(imgPath)
    resizeImg = resizeImageKeepAspectRatio(width = OUTPUT_IMAGE_WIDTH, img = targetImg)
    decreaseRgbOfImg(resizeImg)

    # 画像のリサイズ
    # img[top : bottom, left : right]
    croppedTargetImg = resizeImg[0 : OUTPUT_IMAGE_HEIGHT, 0 : OUTPUT_IMAGE_WIDTH]

    # グレースケール変換
    croppedTargetGrayImg = cv2.cvtColor(croppedTargetImg, cv2.COLOR_BGR2GRAY)
    # 画像の二値化
    thImg = cv2.threshold(croppedTargetGrayImg, 123 , 255, cv2.THRESH_BINARY)[1]
    # 輪郭抽出
    contours, hierarchies = cv2.findContours(thImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectCount = 0
    for rect, hierarchy in zip(contours, hierarchies[0]):
        if not isShowRectangle(rect) or hierarchy[3] == -1:
            continue
        x, y, w, h = cv2.boundingRect(rect)

        if w == RECTANGLE_WIDTH and h == RECTANGLE_HEIGHT:
            rectCount += 1

    modelImg = []
    if rectCount == 0 or rectCount > 6:
        print("Don't match!")
        modelImg = getAppropriatePkmnImg(numberOfParty = 1)
    else:
        modelImg = getAppropriatePkmnImg(rectCount)

    matchRate = round(calcMatchRate(modelImg, croppedTargetImg) * 100, 2)
    print("Rectangle count: {rectCount}".format(rectCount = rectCount))
    print("{matchRate}%".format(matchRate = matchRate))
    cv2.imshow("Comparison with pkmn img", numpy.hstack((modelImg, croppedTargetImg)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
