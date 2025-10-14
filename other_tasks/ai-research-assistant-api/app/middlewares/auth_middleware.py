from jose import jwt, JWTError
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from app.config import settings

EXEMPT_PATHS = {"/api/v1/users/login", "/api/v1/users/register", "/docs", "/openapi.json"}


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header.startswith("Bearer "):
            return JSONResponse({"message": "Unauthorized"}, status_code=status.HTTP_401_UNAUTHORIZED)
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            request.state.user_id = int(payload["sub"])
        except JWTError:
            return JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"message": "Token is invalid."})

        return await call_next(request)
