import logging
import time

from config import ANALYZE_NAME, PHYSX_NAME
from connection.Connection import Connection
from services.SharedMemoryService import read_to_shared_memory, write_to_shared_memory, unpack


class SharedMemoryConnection(Connection):
    def __init__(self):
        self.__physx_data = None

    def set_connection(self):
        pass

    def receive_data(self):
        self.__physx_data, flag = unpack(read_to_shared_memory(PHYSX_NAME))
        while not flag:
            self.__physx_data, flag = unpack(read_to_shared_memory(PHYSX_NAME))
            time.sleep(0.1)

        logging.debug(f"Полученные данные из Physx: {self.__physx_data}")
        return self.__physx_data

    def send_data(self, data):
        logging.debug("Данные сохраняются в AnalyzeSM...")
        write_to_shared_memory(ANALYZE_NAME, data, True)
        write_to_shared_memory(PHYSX_NAME, self.__physx_data, False)
        logging.debug("Данные успешно сохранены в AnalyzeSM!")

    def close_connection(self):
        pass
