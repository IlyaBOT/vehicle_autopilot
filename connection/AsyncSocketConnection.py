import asyncio
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, WebSocket, APIRouter

from config import PORT, HOST, QUESTION_FOR_SIM, CORRECT_ANSWER
from connection.Connection import Connection
from services.GeneralService import create_img


class AsyncSocketConnection(Connection):
    def __init__(self):
        self.__app = FastAPI()
        self.__websocket: Optional[WebSocket] = None

        # отдельная задача для запуска сервера
        self.__server_task: Optional[asyncio.Task] = None

        self.__sending_queue = asyncio.Queue()
        self.__create_route()

    async def set_connection(self):
        """Запуск сервера и ожидание подключения симулятора"""
        logging.info("Сервер запущен")
        logging.info("Ждем запуска симулятора...")

        self.__server_task = asyncio.create_task(self.__run_server())

        while not self.__websocket:
            await asyncio.sleep(1)
        logging.info("Есть соединение с симулятором!")

    async def send_data(self, data: str):
        logging.debug(f"Ответ: {data}")
        await self.__sending_queue.put(data)
        await asyncio.sleep(0.5)

    async def close_connection(self):
        await self.send_data("close_connection")
        logging.info("Разорвали соединение с вебсокетом")
        logging.info("Отключение сервера...")

        try:
            self.__server_task.cancel()
            await self.__server_task
        except asyncio.CancelledError:
            logging.info("Сервер успешно остановлен.")

    def __create_route(self):
        router = APIRouter()
        router.add_api_websocket_route("/", self.__websocket_sim)
        self.__app.include_router(router)

    async def __websocket_sim(self, websocket: WebSocket):
        await websocket.accept()
        await self.__connect_sim(websocket)

        while True:
            new_data = await self.__sending_queue.get()
            await websocket.send_text(new_data)

    async def __connect_sim(self, websocket):
        await websocket.send_text(QUESTION_FOR_SIM)
        answer = await websocket.receive_text()

        if answer != CORRECT_ANSWER:
            logging.info(f"Вебсокет прислал не верный ответ: \"{answer}\"")
            return

        self.__websocket = websocket
        logging.info(f"Симулятор подключен: {websocket}")

    async def __run_server(self):
        server = uvicorn.Server(uvicorn.Config(self.__app, host=HOST, port=PORT, log_level="critical"))
        await server.serve()
        logging.info("Сервер отключен")

    async def get_screen(self):
        await self.__websocket.send_text("getScreen")
        image = await self.__websocket.receive_text()
        create_img(image)

    def receive_data(self):
        pass
