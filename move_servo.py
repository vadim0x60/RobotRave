#!/usr/bin/env python3
"""
Interactive servo tester.
- Type 1-16 for bus servos
- Type p1 or p2 for PWM servos (head)

Usage:
    python3 move_servo.py
"""

import sys
import time

sys.path.insert(0, '/home/pi/TonyPi/HiwonderSDK')
import hiwonder.ros_robot_controller_sdk as rrc

board = rrc.Board()

print("Servo Tester")
print("  1-16  = bus servos (body)")
print("  p1    = PWM servo 1 (head tilt)")
print("  p2    = PWM servo 2 (head pan)")
print("  c     = center all")
print("  q     = quit")
print("-" * 40)

while True:
    try:
        cmd = input("> ").strip().lower()

        if cmd in ['q', 'quit', 'exit']:
            print("Exiting...")
            break
        elif cmd in ['c', 'center']:
            print("Centering all servos...")
            for i in range(1, 17):
                board.bus_servo_set_position(0.3, ((i, 500),))
            board.pwm_servo_set_position(0.3, ((1, 1500), (2, 1500)))
            print("Done")
        elif cmd.startswith('p'):
            # PWM servo
            try:
                pwm_id = int(cmd[1:])
                if pwm_id in [1, 2]:
                    print(f"Wiggling PWM servo {pwm_id}...")
                    # PWM range is typically 500-2500, center 1500
                    board.pwm_servo_set_position(0.4, ((pwm_id, 1200),))
                    time.sleep(0.5)
                    board.pwm_servo_set_position(0.4, ((pwm_id, 1800),))
                    time.sleep(0.5)
                    board.pwm_servo_set_position(0.3, ((pwm_id, 1500),))
                    print("Done")
                else:
                    print("PWM servos are p1 or p2")
            except ValueError:
                print("Use p1 or p2 for PWM servos")
        else:
            try:
                servo_id = int(cmd)
                if 1 <= servo_id <= 16:
                    print(f"Wiggling bus servo {servo_id}...")
                    board.bus_servo_set_position(0.4, ((servo_id, 350),))
                    time.sleep(0.5)
                    board.bus_servo_set_position(0.4, ((servo_id, 650),))
                    time.sleep(0.5)
                    board.bus_servo_set_position(0.3, ((servo_id, 500),))
                    print("Done")
                else:
                    print("Bus servos are 1-16, PWM are p1/p2")
            except ValueError:
                print("Enter 1-16, p1, p2, 'c', or 'q'")
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break
