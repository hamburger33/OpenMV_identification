import sensor, image, time, ustruct
from pyb import UART

last_center_x = 160
x = [0, 0, 0, 0, 0, 0]
rec_x = [0, 0, 0, 0, 0, 0]
x_2 = [0, 0, 0, 0, 0, 0, 0, 0]
clock = time.clock()

threshold_red = (5, 80, 15, 80, 5, 70)
threshold_black = (-20, 20, -20, 20, -20, 20)

uart = None

def init_redline(uart_instance):
    global uart
    uart = uart_instance
    # 不再调用 sensor.reset() 等，全局已在 main 统一初始化
    # 如有需要，可调整白平衡、增益等（可选）
    sensor.set_auto_whitebal(False)
    sensor.set_auto_gain(False)
    # 注意：本模块处理使用 QQVGA（全幅）图像

def sending_data(cx, cy, cw, ch):
    data = ustruct.pack("<bbhhhhb", 0x2C, 0x12,
                         int(cx), int(cy), int(cw), int(ch),
                         0xFF)
    uart.write(data)

def find_max(blobs):
    max_size = 0
    max_blob = None
    for blob in blobs:
        if blob[2] * blob[3] > max_size:
            max_blob = blob
            max_size = blob[2] * blob[3]
    return max_blob

def redline_follow():
    global last_center_x, x, rec_x, x_2
    clock.tick()
    img = sensor.snapshot()

    # X方向切割检测红线
    img_green = img.copy()
    center = 0
    blob_num = 0
    for n in range(0, 6):
        blobs = img_green.find_blobs([threshold_red], roi=(0, n * 20, 160, 10),
                                      pixels_threshold=50, area_threshold=50, merge=True)
        if blobs:
            max_blob = find_max(blobs)
            if max_blob:
                img.draw_rectangle(max_blob.rect(), color=(0, 255, 0))
                img.draw_cross(max_blob.cx(), max_blob.cy(), color=(0, 255, 0))
                x[n] = max_blob.cx()
                rec_x[n] = max_blob[2]
                center += x[n]
                blob_num += 1

    # Y方向切割检测红线
    img_blue = img.copy()
    center_2 = 0
    blob_num_y = 0
    for n in range(0, 8):
        blobs_y = img_blue.find_blobs([threshold_red], roi=(n * 20, 0, 20, 120),
                                      pixels_threshold=100, area_threshold=100, merge=True)
        if blobs_y:
            max_blob_y = find_max(blobs_y)
            if max_blob_y:
                img.draw_rectangle(max_blob_y.rect(), color=(0, 0, 255))
                img.draw_cross(max_blob_y.cx(), max_blob_y.cy(), color=(0, 0, 255))
                x_2[n] = max_blob_y.cx()
                center_2 += x_2[n]
                blob_num_y += 1

    # 检测黑色终点线
    img_black = img.copy()
    blobs_black = img_black.find_blobs([threshold_black], pixels_threshold=50,
                                        area_threshold=50, merge=True)
    max_blob_black = (0, 0, 0, 0)
    if blobs_black:
        max_blob_black = find_max(blobs_black)
        if max_blob_black:
            img.draw_rectangle(max_blob_black.rect(), color=(255, 255, 0))

    # 判断直角标志
    right_angle = 0
    for n in range(0, 6):
        if rec_x[n] >= 70:
            right_angle = 1

    # 数据发送判断
    if max_blob_black[2] > 130 and blob_num >= 3:
        info = bytearray([0x2C, 0x12, 9, 9, 9, 9, 0xFF])
        uart.write(info)
        print("Stop")
        return 11  # 继续保持红线跟踪模式
    else:
        if blob_num >= 6:
            center_x = int(center / blob_num)
            last_center_x = center_x
            if right_angle:
                print("直角弯，center_x =", center_x)
                info = bytearray([0x2C, 0x12, center_x, blob_num, x[5], 2, 0xFF])
                uart.write(info)
            else:
                print("直走或小弯，center_x =", center_x)
                info = bytearray([0x2C, 0x12, center_x, blob_num, x[5], 1, 0xFF])
                uart.write(info)
        elif blob_num < 6 and blob_num > 0 and blob_num_y > 0:
            center_x = int(center_2 / blob_num_y)
            last_center_x = center_x
            if right_angle:
                print("直角弯，center_x =", center_x)
                info = bytearray([0x2C, 0x12, center_x, blob_num_y, x[5], 2, 0xFF])
                uart.write(info)
            else:
                print("拐弯，center_x =", center_x)
                info = bytearray([0x2C, 0x12, center_x, blob_num_y, x[5], 1, 0xFF])
                uart.write(info)
        else:
            info = bytearray([0x2C, 0x12, last_center_x, 1, 1, 1, 0xFF])
            uart.write(info)
            print("last_center_x =", last_center_x)
    return 11  # 返回模式标识（继续红线跟踪）

if __name__ == "__main__":
    # 为测试时单独运行时使用（已统一在 main 中初始化，则不建议直接调用）
    init_redline(UART(3,115200))
    while True:
        redline_follow()
        time.sleep_ms(50)
