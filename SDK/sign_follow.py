import sensor, image, time, ml, math, uos, gc,ustruct
from pyb import UART

sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((240, 240))       # Set 240x240 window.
sensor.skip_frames(20)          # Let the camera adjust.
uart = UART(3, 115200)
uart.init(115200, bits=8, parity=None, stop=1)

net = None
labels = None
min_confidence = 0.5

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = ml.Model("trained.tflite", load_to_fb=uos.stat('trained.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

colors = [ # Add more colors if you are detecting more than 7 types of classes at once.
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

threshold_list = [(math.ceil(min_confidence * 255), 255)]

def fomo_post_process(model, inputs, outputs):
    ob, oh, ow, oc = model.output_shape[0]

    x_scale = inputs[0].roi[2] / ow
    y_scale = inputs[0].roi[3] / oh

    scale = min(x_scale, y_scale)

    x_offset = ((inputs[0].roi[2] - (ow * scale)) / 2) + inputs[0].roi[0]
    y_offset = ((inputs[0].roi[3] - (ow * scale)) / 2) + inputs[0].roi[1]

    l = [[] for i in range(oc)]

    for i in range(oc):
        img = image.Image(outputs[0][0, :, :, i] * 255)
        blobs = img.find_blobs(
            threshold_list, x_stride=1, y_stride=1, area_threshold=1, pixels_threshold=1
        )
        for b in blobs:
            rect = b.rect()
            x, y, w, h = rect
            score = (
                img.get_statistics(thresholds=threshold_list, roi=rect).l_mean() / 255.0
            )
            x = int((x * scale) + x_offset)
            y = int((y * scale) + y_offset)
            w = int(w * scale)
            h = int(h * scale)
            l[i].append((x, y, w, h, score))
    return l

def sending_data(cx, cy, cw, ch):
    global uart
    data = ustruct.pack("<bbhhhhb", 0x2C, 0x12,
                         int(cx), int(cy),
                         int(cw), int(ch),
                         0xFF)
    uart.write(data)

flag = cmd = 0
clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot()

    for i, detection_list in enumerate(net.predict([img], callback=fomo_post_process)):
        if i == 0: continue  # background class
        if len(detection_list) == 0: continue  # no detections for this class?

        #print("%s" % labels[i])
        sign = labels[i]
        for x, y, w, h, score in detection_list:
            center_x = math.floor(x + (w / 2))
            center_y = math.floor(y + (h / 2))
            #print(f"x {center_x}\ty {center_y}\tscore {score}")
            img.draw_circle((center_x, center_y, 12), color=colors[i])
            flag = 1

    info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
    if uart.any():
        data = uart.read(1)
        if data and len(data) == 1:
            cmd = data[0]
    if(cmd == 0):
        if(flag):
            if(sign == "forward"):
                info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
                uart.write(info)
                print("run forward")
                gc.collect()
            elif(sign == "left"):
                info = bytearray([0x2C, 0x12, 2, 2, 2, 2, 0xFF])
                uart.write(info)
                print("turn left")
                gc.collect()
            elif(sign == "right"):
                info = bytearray([0x2C, 0x12, 3, 3, 3, 3, 0xFF])
                uart.write(info)
                print("turn right")
                gc.collect()
            elif(sign == "uturn"):
                info = bytearray([0x2C, 0x12, 4, 4, 4, 4, 0xFF])
                uart.write(info)
                print("turn back")
                gc.collect()
            elif(sign == "park"):
                info = bytearray([0x2C, 0x12, 5, 5, 5, 5, 0xFF])
                uart.write(info)
                print("Park")
                gc.collect()
        else:
            info = bytearray([0x2C, 0x12, 0, 0, 0, 0, 0xFF])
            uart.write(info)
            print("none")
    else:
        info = bytearray([0x2C, 0x12, cmd, cmd, cmd, cmd, 0xFF])
        uart.write(info)
        print("%d" % cmd)
