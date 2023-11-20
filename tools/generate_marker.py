import argparse
import cv2
import numpy as np


INCH2MM = 25.4
BG_COLOR = (255, 255, 255)
PATTERN_COLOR = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(prog='Marker image generator')
    parser.add_argument('--display_x_res', type=int, default=1080)
    parser.add_argument('--display_y_res', type=int, default=2400)
    parser.add_argument('--display_ppi', type=float, default=421)
    parser.add_argument('--x_cells', type=int, default=7)
    parser.add_argument('--y_cells', type=int, default=10)
    parser.add_argument('--cell_size_mm', type=float, default=8)
    parser.add_argument('--output', type=str, default='pattern.png')
    return parser.parse_args()


def generate(args):
    pixel_size_mm = INCH2MM / args.display_ppi

    x_center_idx = args.display_x_res // 2
    x_center_mm = x_center_idx * pixel_size_mm

    y_center_idx = args.display_y_res // 2
    y_center_mm = y_center_idx * pixel_size_mm

    output = np.zeros(
        shape=(args.display_y_res, args.display_x_res, 3),
        dtype=np.uint8
    )
    output[:, :] = BG_COLOR

    pattern_x_size_mm = args.x_cells * args.cell_size_mm
    x_offset_pixels = x_center_idx - int(0.5 * pattern_x_size_mm / pixel_size_mm)
    pattern_y_size_mm = args.y_cells * args.cell_size_mm
    y_offset_pixels = y_center_idx - int(0.5 * pattern_y_size_mm / pixel_size_mm)

    pattern_x_size_pixels = int(np.ceil(pattern_x_size_mm / pixel_size_mm))
    pattern_y_size_pixels = int(np.ceil(pattern_y_size_mm / pixel_size_mm))

    for y_idx in range(pattern_y_size_pixels):
        y_pos = y_idx + y_offset_pixels
        if y_pos < 0 or y_pos >= args.display_y_res:
            continue
        y_cell_idx = int(y_idx * pixel_size_mm / args.cell_size_mm)
        y_even = y_cell_idx % 2 == 0
        
        for x_idx in range(pattern_x_size_pixels):
            x_pos = x_idx + x_offset_pixels
            if x_pos < 0 or x_pos >= args.display_x_res:
                continue              
            x_cell_idx = int(x_idx * pixel_size_mm / args.cell_size_mm)
            x_even = x_cell_idx % 2 == 0

            if not x_even ^ y_even:
                output[y_pos, x_pos] = PATTERN_COLOR

    cv2.imwrite(args.output, output)
    print(f'Pattern {output.shape} has been saved @ {args.output}')


if __name__ == '__main__':
    generate(parse_args())
