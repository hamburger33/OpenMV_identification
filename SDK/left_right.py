import sensor, image, time, ml, math, uos, gc
from pyb import UART

net = None
labels = None
uart = UART(3, 115200)
uart.init(115200, bits=8, parity=None, stop=1)

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

# 加载两个模型和标签
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
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1)
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

clock = time.clock()
def sign_follow():
    sign_counter = {}  # 记录每个标志的连续识别次数
    prev_sign = None   # 上一次识别的标志
    while True:
        clock.tick()
        img = sensor.snapshot()
        cmd = 0
        sign = None

        # 最佳识别结果初始化
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
                # 如果不是 left/right，才考虑用 other 模型结果
                if label not in ["left", "right"] and score > best_score:
                    best_score = score
                    best_det = (x, y, w, h)
                    best_label = label
                    best_color = colors[i]

        # ------ 显示结果 ------
        if best_det:
            x, y, w, h = best_det
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            #print(">> Label:", best_label)
            #print(f"x: {center_x}\ty: {center_y}\tscore: {best_score}")
            img.draw_circle((center_x, center_y, 12), color=best_color)
            img.draw_string(center_x + 15, center_y - 10, best_label, color=best_color)
            img.draw_string(center_x + 15, center_y + 10, "S:{:.2f}".format(best_score), color=best_color)
            sign = best_label

        if sign is None:
            # 如果检测结果为 None 且之前的不为 None，则重置计数
            if prev_sign is not None:
                sign_counter.clear()
            prev_sign = None
        else:
            if sign == prev_sign:
                sign_counter[sign] = sign_counter.get(sign, 0) + 1
            else:
                sign_counter[sign] = 1
            prev_sign = sign

        # ------ 根据 UART 命令和检测结果发送信息 ------
        if cmd == 0:
            # 若检测不到任何有效标志，则 sign 为 None
            if sign is None:
                info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
                uart.write(info)
                print("None，巡线！")
            else:
                # 若检测到标志，但连续计数未达到3次，则依然输出 None
                if sign_counter.get(sign, 0) < 2:
                    info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
                    uart.write(info)
                    print("None，巡线！")
                else:
                    # 连续识别次数大于等于3时，发送对应指令
                    if sign == "forward":
                        info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
                        uart.write(info)
                        print("run forward")
                    elif sign == "left":
                        info = bytearray([0x2C, 0x12, 2, 2, 2, 2, 0xFF])
                        uart.write(info)
                        print("turn left")
                    elif sign == "right":
                        info = bytearray([0x2C, 0x12, 3, 3, 3, 3, 0xFF])
                        uart.write(info)
                        print("turn right")
                    elif sign == "uturn":
                        info = bytearray([0x2C, 0x12, 4, 4, 4, 4, 0xFF])
                        uart.write(info)
                        print("turn back")
                    elif sign == "park":
                        info = bytearray([0x2C, 0x12, 5, 5, 5, 5, 0xFF])
                        uart.write(info)
                        print("Park")
                    gc.collect()
        else:
            info = bytearray([0x2C, 0x12, cmd, cmd, cmd, cmd, 0xFF])
            uart.write(info)
            print("%d" % cmd)

# 如果直接运行此模块用于调试，则初始化后连续执行 sign_follow
if __name__ == "__main__":
    init_sign()
    while True:
        sign_follow()
        time.sleep_ms(50)
