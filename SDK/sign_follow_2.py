import sensor, image, time, ml, math, uos, gc, ustruct
from pyb import UART
from SDK import redline_follow

net = None
labels = None
uart = UART(3, 115200)
uart.init(115200, bits=8, parity=None, stop=1)
clock = time.clock()
delay_time = 3000

def init_sign():
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing((240, 240))
    sensor.skip_frames(20)

min_confidence = 0.5
threshold_list = [(math.ceil(min_confidence * 255), 255)]

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]

def load_model_and_labels(model_path, label_path):
    model = ml.Model(model_path, load_to_fb=uos.stat(model_path)[6] > (gc.mem_free() - (64 * 1024)))
    with open(label_path) as f:
        label_list = [line.strip() for line in f]
    return model, label_list

model_left, labels_left = load_model_and_labels("trained_left.tflite", "labels_left.txt")
model_other, labels_other = load_model_and_labels("trained.tflite", "labels.txt")

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]
    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh
    scale = min(x_scale, y_scale)
    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]
    l = [[] for _ in range(oc)]
    for i in range(oc):
        img_temp = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img_temp.find_blobs(threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1)
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = img_temp.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

def sign_follow():
    sign_counter = {}  # 记录每个标志的连续识别次数
    prev_sign = None   # 上一次识别的标志
    while True:
        clock.tick()
        img = sensor.snapshot()
        cmd = 0
        sign = None
        # if uart.any():
        #     data = uart.read(1)
        #     if data and len(data) == 1:
        #         cmd = data[0]

        best_det = None
        best_label = None
        best_score = 0
        best_color = (255, 255, 255)

        # ------ 1. 使用 LEFT 模型检测 ------
        detections_left = model_left.predict([img], callback=fomo_post_process)
        for i, dets in enumerate(detections_left):
            if i == 0 or len(dets) == 0:
                continue
            for det in dets:
                x, y, w, h, score = det
                if score > best_score:
                    best_score = score
                    best_det = (x, y, w, h)
                    best_label = labels_left[i]
                    best_color = colors[i]

        # ------ 2. 使用 OTHER 模型检测 ------
        detections_other = model_other.predict([img], callback=fomo_post_process)
        for i, dets in enumerate(detections_other):
            if i == 0 or len(dets) == 0:
                continue
            for det in dets:
                x, y, w, h, score = det
                label = labels_other[i]
                if label not in ["left", "right"] and score > best_score:
                    best_score = score
                    best_det = (x, y, w, h)
                    best_label = label
                    best_color = colors[i]

        if best_det:
            x, y, w, h = best_det
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            img.draw_circle((center_x, center_y, 12), color=best_color)
            img.draw_string(center_x + 15, center_y - 10, best_label, color=best_color)
            img.draw_string(center_x + 15, center_y + 10, "S:{:.2f}".format(best_score), color=best_color)
            sign = best_label

        if sign is None:
            if prev_sign is not None:
                sign_counter.clear()
            prev_sign = None
        else:
            if sign == prev_sign:
                sign_counter[sign] = sign_counter.get(sign, 0) + 1
            else:
                sign_counter[sign] = 1
            prev_sign = sign

        if cmd == 0:
            if sign is None:
                # info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
                # uart.write(info)
                redline_follow.redline_follow() #没检测到路标就巡线走
                print("None，巡线！")
            else:
                if sign_counter.get(sign, 0) < 2:
                    # info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
                    # uart.write(info)
                    redline_follow.redline_follow() #没检测到路标就巡线走
                    print("None，巡线！")
                else:
                    if sign == "forward":
                        info = bytearray([0x2C, 0x12, 1, 1, 1, 1, 0xFF])
                        uart.write(info)
                        print("run forward")
                        #pyb.delay(delay_time)  # 延时3000毫秒（2秒）
                    elif sign == "left":
                        info = bytearray([0x2C, 0x12, 2, 2, 2, 2, 0xFF])
                        uart.write(info)
                        print("turn left")
                        #pyb.delay(delay_time)  # 延时3000毫秒（2秒）
                    elif sign == "right":
                        info = bytearray([0x2C, 0x12, 3, 3, 3, 3, 0xFF])
                        uart.write(info)
                        print("turn right")
                        #pyb.delay(delay_time)  # 延时3000毫秒（2秒）
                    elif sign == "uturn":
                        info = bytearray([0x2C, 0x12, 4, 4, 4, 4, 0xFF])
                        uart.write(info)
                        print("turn back")
                        #pyb.delay(delay_time)  # 延时3000毫秒（2秒）
                    elif sign == "park":
                        info = bytearray([0x2C, 0x12, 5, 5, 5, 5, 0xFF])
                        uart.write(info)
                        print("Park")
                        #pyb.delay(delay_time)  # 延时3000毫秒（2秒）
                    gc.collect()
        else:
            info = bytearray([0x2C, 0x12, cmd, cmd, cmd, cmd, 0xFF])
            uart.write(info)
            print("%d" % cmd)

        time.sleep_ms(50)

if __name__ == "__main__":
    init_sign()
    sign_follow()
