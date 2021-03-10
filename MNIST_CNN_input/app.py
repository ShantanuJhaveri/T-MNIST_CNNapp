import cv2 as cv
import numpy as np
from MNIST_CNN_input.model import NN

CNN = NN()

# create a matrix for our drawings
canvas = np.ones((600, 600), dtype="uint8") * 255

# create canvas to be drawn in
canvas[100:500, 100:500] = 0

start_point = None
end_point = None
is_drawing = False


def draw_line(img, start_at, end_at):
    cv.line(img, start_at, end_at, 255, 15)


def on_mouse_events(event, x, y, flags, params):
    global start_point
    global end_point
    global canvas
    global is_drawing
    if event == cv.EVENT_LBUTTONDOWN:
        if is_drawing:
            start_point = (x, y)
    elif event == cv.EVENT_MOUSEMOVE:
        if is_drawing:
            end_point = (x, y)
            draw_line(canvas, start_point, end_point)
            start_point = end_point
    elif event == cv.EVENT_LBUTTONUP:
        is_drawing = False


cv.namedWindow("Test Canvas")
cv.setMouseCallback("Test Canvas", on_mouse_events)

while (True):
    cv.imshow("Test Canvas", canvas)
    key = cv.waitKey(1) & 0xFF

    # this is how i did input commands, pls let me know if you can do something more effecient
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_drawing = True
    elif key == ord('c'):
        # clear canvas
        canvas[100:500, 100:500] = 0

    elif key == ord('p'):
        # set image to the canvas and pass the canvas into the mdoel
        image = canvas[100:500, 100:500]
        result = CNN.predict(image)
        print("PREDICTION : ", result)

cv.destroyAllWindows()

# IDK what the XLA devices are, they are supposed to be some experimental compiler, will have to look into later
