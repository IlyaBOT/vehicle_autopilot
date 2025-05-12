import base64
import logging


def create_img(byte_img: str):
    print(byte_img)
    split_img = byte_img.split("base64")
    if len(split_img) < 2:
        logging.critical("Не корректные данные для преобразования в картинку")
        return

    img = base64.b64decode(split_img[1])

    with open("output.webp", "wb") as img_file:
        img_file.write(img)
