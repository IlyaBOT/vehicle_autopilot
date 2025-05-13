import asyncio

from algorithm.PID import analyze
from config import LOG_LEVEL
from connection.SocketConnection import SocketConnection
from connection.AsyncSocketConnection import AsyncSocketConnection
from connection.SharedMemoryConnection import SharedMemoryConnection
from services.logger import set_logger_config
from vehicle.Vehicle import Vehicle  
from vehicle.my_vehicle_control import control_vehicle

async def start():
    connection = SocketConnection()
    vehicle = Vehicle(connection)
    await connection.set_connection()

    connection_status = connection.receive_data()
    print("connection status:", connection_status)

    await control_vehicle(vehicle, connection)

if __name__ == "__main__":
    asyncio.run(start())
