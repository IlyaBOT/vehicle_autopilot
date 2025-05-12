import mmap
import struct

from config import STRUCT_SIZE


def read_to_shared_memory(sh_name: str):
    with mmap.mmap(-1, STRUCT_SIZE, sh_name, access=mmap.ACCESS_READ) as mm:
        # Чтение 4 double и 1 булевского флага
        data = struct.unpack('4d?', mm)
        return data


def write_to_shared_memory(sh_name: str, new_values, new_flag: bool):
    with mmap.mmap(-1, STRUCT_SIZE, sh_name, access=mmap.ACCESS_WRITE) as mm:
        # Упаковываем новые данные в формат '4d?' (4 double и 1 bool)
        mm.seek(0)
        mm.write(struct.pack('4d?', *new_values, new_flag))


def unpack(values):
    return values[:4], values[-1]
