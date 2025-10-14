import asyncio
import grpc
from app.proto import user_pb2_grpc
from app.grpc_handlers import UserService
from app.database import engine, Base


async def serve():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    server = grpc.aio.server()
    user_pb2_grpc.add_UserServiceServicer_to_server(UserService(), server)
    server.add_insecure_port('[::]:50051')
    print("Starting server on port 50051")
    await server.start()
    await server.wait_for_termination()


if __name__ == '__main__':
    asyncio.run(serve())