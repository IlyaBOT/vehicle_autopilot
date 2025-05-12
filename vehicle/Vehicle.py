import json
from connection.SocketConnection import SocketConnection

class Vehicle:
    def __init__(self, __connection: SocketConnection):
        self.__connection = __connection
    
    # первый параметр - мощность вращения левых колес в процентах, второй параметр - мощность вращения правых колес.
    # если значение положительное, то колесо вращается по часовой стрелке, если отрицательно, то против часовой стрелки.
    def setMotorPower(self, right, left):
        self.__connection.send_data(json.dumps({'name': 'setMotorPower', 'payload': {
                'right': right, 
                'left': left,
            }}))
        
        status = self.__connection.receive_data()

        # print('status:', status)

    # метод повората робота относительно текущего положения.Отрицательные значения параметра angle будут поворачивать робота по часовой стрелке, положительные - против часовой стрелки
    def rotate(self, angle):
        self.__connection.send_data(json.dumps({'name': 'rotate', 'payload': {
                'angle': angle, 
            }}))
        
        status = self.__connection.receive_data()

        # print('status:', status)