import cv2

class Visualizer_Global():
    def __init__(self):
        pass

    def showImg(self, img, name='asdf'):
        cv2.imshow(name, img)
        cv2.waitKey(10000)

    def drawBBoxCenter(self, img, offset, relX, relY):
        x = offset[:, 0, None] + relX
        y = offset[:, 1, None] + relY
        N = offset.shape[0]
        for i in range(0, N):
            img = cv2.circle(img, (x[i], y[i]), 4, (255, 0, 0), -1)
        return img

    def drawBBox(self, img, bboxes, objNames):
        if bboxes is None:
            return img
        for i in range(0, bboxes.shape[0]):
            x, y = bboxes[i, 0], bboxes[i, 1]
            x2, y2 = x + bboxes[i, 2], y + bboxes[i, 3]
            img = cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
            img = cv2.putText(img, objNames[i], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        return img