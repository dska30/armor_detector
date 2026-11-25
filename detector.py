import cv2
import numpy as np

class ArmorDetector:
    def __init__(self, img):
        # 参数
        self.thresh_color_diff = 60
        self.kernel = np.ones((3, 3), np.uint8)
        self.min_area = 0.1
        self.min_ratio = 2.5
        self.max_ratio = 50000

        self.angle_tol = 15
        self.max_slope = 0.45
        self.min_dist = 18
        self.max_dist = 10000
        self.max_len_ratio = 1.6

        self.img = img.copy()
        self.all_lights = []
        self.armor_list = []
        self.mask_red = None        # 创建对象自动执行整套识别
        self.mask_blue = None

        self._run_detect()

    def _get_light_list(self, mask, color):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        light_list = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            box = cv2.boxPoints(rect)
            box = box.astype(int)
            long_side = max(w, h)
            short_side = min(w, h)
            ratio = long_side / short_side
            if ratio > self.max_ratio or ratio < self.min_ratio:
                continue
            if angle < -45:
                angle += 90
            light_list.append({
                "cx": cx, "cy": cy, "w": w, "h": h,
                "angle": angle, "color": color, "box": box
            })
        return light_list

    def _is_pair(self, light1, light2):
        if light1["color"] != light2["color"]:
            return False
        angle_diff = abs(light1["angle"] - light2["angle"])
        if angle_diff > self.angle_tol:
            return False
        dy = abs(light1["cy"] - light2["cy"])
        dx = abs(light1["cx"] - light2["cx"])
        if dy / dx > self.max_slope:
            return False
        dist = np.hypot(light1["cx"] - light2["cx"], light1["cy"] - light2["cy"])
        if dist < self.min_dist or dist > self.max_dist:
            return False
        len1 = max(light1["w"], light1["h"])
        len2 = max(light2["w"], light2["h"])
        len_ratio = max(len1, len2) / (min(len1, len2) + 1e-6)
        if len_ratio > self.max_len_ratio:
            return False
        return True

    def _run_detect(self):         #主方法
        #分离红蓝通道
        b, g, r = cv2.split(self.img)
        red_diff = cv2.subtract(r, g)
        blue_diff = cv2.subtract(b, g)
        #二值化
        _, self.mask_red = cv2.threshold(red_diff, self.thresh_color_diff, 255, cv2.THRESH_BINARY)
        _, self.mask_blue = cv2.threshold(blue_diff, self.thresh_color_diff, 255, cv2.THRESH_BINARY)
        #闭开运算保主体
        self.mask_red = cv2.morphologyEx(self.mask_red, cv2.MORPH_CLOSE, self.kernel)
        self.mask_red = cv2.morphologyEx(self.mask_red, cv2.MORPH_OPEN, self.kernel)
        self.mask_blue = cv2.morphologyEx(self.mask_blue, cv2.MORPH_CLOSE, self.kernel)
        self.mask_blue = cv2.morphologyEx(self.mask_blue, cv2.MORPH_OPEN, self.kernel)
        #调用灯条方法，拿到红蓝灯条
        red_lists = self._get_light_list(self.mask_red, "red")
        blue_lists = self._get_light_list(self.mask_blue, "blue")
        self.all_lights = red_lists + blue_lists
        #调用配对方法，拿到装甲板
        self.armor_list = []
        used = [False] * len(self.all_lights)
        for i in range(len(self.all_lights)):
            if not used[i]:
                for j in range(i + 1, len(self.all_lights)):
                    if used[j]:
                        continue
                    l1 = self.all_lights[i]
                    l2 = self.all_lights[j]
                    if self._is_pair(l1, l2):
                        box1 = l1["box"]
                        box2 = l2["box"]
                        all_pts = np.concatenate([box1, box2])
                        armor_rect = cv2.minAreaRect(all_pts)
                        armor_cx, armor_cy = armor_rect[0]
                        armor_box = cv2.boxPoints(armor_rect)
                        armor_box = armor_box.astype(int)
                        self.armor_list.append((armor_cx, armor_cy, l1["color"], armor_box))
                        used[i] = True
                        used[j] = True
                        break

    def draw_result(self, our_color="blue"):
        img_copy = self.img.copy()
        # 画灯条
        for light in self.all_lights:
            box = light["box"]
            if light["color"] == "red":
                cv2.drawContours(img_copy, [box], 0, (255, 0, 0), 5)
            else:
                cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 5)
        # 画装甲
        for cx, cy, c, box in self.armor_list:
            if c == our_color:
                cv2.drawContours(img_copy, [box], 0, (0, 0, 255), 2)
                cv2.circle(img_copy, (int(cx), int(cy)), 10, (0, 255, 0), -1)
                text = f"({cx:.0f}),({cy:.0f})"
                cv2.putText(img_copy, text, (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
            else:
                cv2.drawContours(img_copy, [box], 0, (255, 0, 0), 2)
                cv2.circle(img_copy, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                text = f"({cx:.0f}),({cy:.0f})"
                cv2.putText(img_copy, text, (int(cx)+8, int(cy)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 1)
        return img_copy


if __name__ == "__main__":
    path = "//输入你的图片地址"
    img = cv2.imread(path)
    detector = ArmorDetector(img)

    img_copy = detector.draw_result(our_color="blue")

    cv2.namedWindow("window_origin", cv2.WINDOW_NORMAL)
    cv2.namedWindow("window_red_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("window_blue_mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    cv2.imshow("window_origin", img)
    cv2.imshow("window_red_mask", detector.mask_red)
    cv2.imshow("window_blue_mask", detector.mask_blue)
    cv2.imshow("result", img_copy)
    cv2.waitKey(0)