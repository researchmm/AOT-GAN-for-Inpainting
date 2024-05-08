import cv2


class Sketcher:
    def __init__(self, windowname, dests, colors_func, thick, type):
        self.prev_pt = None
        self.windowname = windowname
        self.dests = dests
        self.colors_func = colors_func
        self.dirty = False
        self.show()
        self.thick = thick
        if type == "bbox":
            cv2.setMouseCallback(self.windowname, self.on_bbox)
        else:
            cv2.setMouseCallback(self.windowname, self.on_mouse)

    def large_thick(
        self,
    ):
        self.thick = min(48, self.thick + 1)

    def small_thick(
        self,
    ):
        self.thick = max(3, self.thick - 1)

    def show(self):
        cv2.imshow(self.windowname, self.dests[0])

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.line(dst, self.prev_pt, pt, color, self.thick)
            self.dirty = True
            self.prev_pt = pt
            self.show()

    def on_bbox(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            for dst, color in zip(self.dests, self.colors_func()):
                cv2.rectangle(dst, self.prev_pt, pt, color, -1)
            self.dirty = True
            self.prev_pt = None
            self.show()
