import grpc
from app.proto import user_pb2, user_pb2_grpc
from app.crud import create_user, update_user, get_user, delete_user, get_users
from app.database import AsyncSessionLocal
from app.schemas import UserCreate

USER_NOT_FOUND_MSG = "User not found"

class UserService(user_pb2_grpc.UserServiceServicer):
    async def CreateUser(self, request, context):
        async with AsyncSessionLocal() as db:
            user_info = UserCreate(name=request.name, email=request.email)
            user = await create_user(db, user_info)
            return user_pb2.UserResponse(id=user.id, name=user.name, email=user.email)

    async def GetUser(self, request, context):
        async with AsyncSessionLocal() as db:
            user = await get_user(db, request.id)
            if not user:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(USER_NOT_FOUND_MSG)
                return user_pb2.UserResponse()
            return user_pb2.UserResponse(id=user.id, name=user.name, email=user.email)

    async def ListUsers(self, request, context):
        async with AsyncSessionLocal() as db:
            users = await get_users(db)
            return user_pb2.UserList(
                users=[
                    user_pb2.UserResponse(id=user.id, name=user.name, email=user.email)
                    for user in users
                ]
            )

    async def UpdateUser(self, request, context):
        async with AsyncSessionLocal() as db:
            user_info = UserCreate(name=request.name, email=request.email)
            user = await update_user(db, request.id, user_info)
            if not user:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(USER_NOT_FOUND_MSG)
                return user_pb2.UserResponse()
            return user_pb2.UserResponse(id=user.id, name=user.name, email=user.email)

    async def DeleteUser(self, request, context):
        async with AsyncSessionLocal() as db:
            success = await delete_user(db, request.id)
            if not success:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(USER_NOT_FOUND_MSG)
            return user_pb2.DeleteRes_
