import sensor, time, ustruct
from pyb import UART, LED
from SDK import redline_follow, sign_follow, sign_follow_2
def main():
    uart = UART(3, 115200)
    uart.init(115200, bits=8, parity=None, stop=1)
    #LED(2).on()
    current_mode = None
    command = 00
    redline_follow.init_redline()
    sign_follow.init_sign()
    sign_follow_2.init_sign_2()
    while True:
        data = uart.read(1)
        if data is not None and len(data) == 1:
            command = data[0]
            print("Received:", command)
        if command != current_mode:
            if command == 11:
                print("切换到红线跟踪模式")
                current_mode = 11
            elif command == 22:
                print("切换到标志跟踪模式")
                current_mode = 22
            elif command == 44:
                print("切换到标志+红线跟踪模式")
                current_mode = 44
            else:
                current_mode = None
        if current_mode == 11:
            current_mode = redline_follow.redline_follow()
        elif current_mode == 22:
            current_mode = sign_follow.sign_follow()
        elif current_mode == 44:
            current_mode == sign_follow_2.sign_follow_2()
        else:
            img = sensor.snapshot()
            time.sleep_ms(10)
if __name__ == "__main__":
    main()
