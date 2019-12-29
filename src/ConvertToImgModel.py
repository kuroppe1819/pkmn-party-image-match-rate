import cv2
import sys
import glob
import numpy

OUTPUT_IMAGE_WIDTH = 480
THRESHOLD_CONTOUR_AREA = 500

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

if __name__ == "__main__":
    if len(sys.argv) == 1: # 引数がファイル名のみ
        raise Exception("引数が不正です")
    sys.argv.pop(0) # pythonのプログラムファイルを削除
    imgModelPathList = [path for arg in sys.argv for path in glob.glob(arg)]

    if len(imgModelPathList) == 0:
        sys.exit()

    croppedPkmnPartyImgs = []
    for i, imgPath in enumerate(imgModelPathList):
        pkmnImg = cv2.imread(imgPath)
        resizeImg = resizeImageKeepAspectRatio(width = OUTPUT_IMAGE_WIDTH, img = pkmnImg)
        decreaseRgbOfImg(resizeImg)

        # 画像のリサイズ
        # img[top : bottom, left : right]
        resizeImgHeight = resizeImg.shape[0]
        footerTopY = int(resizeImgHeight * 4 / 5)
        croppedPkmnPartyImgs.append(resizeImg[0 : footerTopY, 0 : OUTPUT_IMAGE_WIDTH])
    numpy.save("./output/model/pkmn_party_imgs_model.npy", croppedPkmnPartyImgs)
