import cv2


def main():
    while True:
        try:
            image = cv2.imread('Image.png')
            cv2.imshow("Image", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()
