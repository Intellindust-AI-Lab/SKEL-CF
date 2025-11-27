from typing import List
def float_to_int8(color: List[float]) -> List[int]:
    return [int(c * 255) for c in color]

def int8_to_float(color: List[int]) -> List[float]:
    return [c / 255 for c in color]

def int8_to_hex(color: List[int]) -> str:
    return '#%02x%02x%02x' % tuple(color)

def float_to_hex(color: List[float]) -> str:
    return int8_to_hex(float_to_int8(color))

def hex_to_int8(color: str) -> List[int]:
    return [int(color[i+1:i+3], 16) for i in (0, 2, 4)]

def hex_to_float(color: str) -> List[float]:
    return int8_to_float(hex_to_int8(color))


# TODO: incorporate https://github.com/vye16/slahmr/blob/main/slahmr/vis/colors.txt

class ColorPalette:

    # Picked from: https://colorsite.librian.net/
    presets = {
        'black'        : '#2b2b2b',
        'white'        : '#eaedf7',
        'pink'         : '#e6cde3',
        'light_pink'   : '#fdeff2',
        'blue'         : '#89c3eb',
        'purple'       : '#a6a5c4',
        'light_purple' : '#bbc8e6',
        'red'          : '#d3381c',
        'orange'       : '#f9c89b',
        'light_orange' : '#fddea5',
        'brown'        : '#b48a76',
        'human_yellow' : '#f1bf99',
        'orange_red'   : '#FF7F50',
        'green'        : '#a8c97f',
    }

    presets_int8 = {k: hex_to_int8(v) for k, v in presets.items()}
    presets_float = {k: int8_to_float(v) for k, v in presets_int8.items()}
    presets_hex = {k: v for k, v in presets.items()}