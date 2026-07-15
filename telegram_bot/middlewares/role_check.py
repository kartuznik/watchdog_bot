"""Role-based middleware and decorator for Telegram command access."""

from __future__ import annotations

from functools import wraps
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message

from agents.roles import RoleName, get_role, has_required_role

COMMAND_ROLE_REQUIREMENTS: dict[str, RoleName] = {
    "/setadmin": "owner",
    "/removeadmin": "owner",
    "/admins": "owner",
    "/restart": "owner",
    "/selftest": "admin",
    "/fulldiag": "admin",
    "/status": "admin",
}


def _extract_command(message: Message) -> str:
    if not message.text:
        return ""
    first_token = message.text.strip().split(maxsplit=1)[0].lower()
    return first_token


def require_role(required_role: RoleName):
    """Decorator for handlers that require minimum role."""

    def decorator(handler: Callable[..., Awaitable[Any]]):
        @wraps(handler)
        async def wrapper(message: Message, *args: Any, **kwargs: Any):
            user_id = message.from_user.id if message.from_user else 0
            if not has_required_role(user_id, required_role):
                current = get_role(user_id)
                await message.answer(
                    f"Недостаточно прав. Нужна роль `{required_role}`, твоя роль: `{current}`."
                )
                return None
            return await handler(message, *args, **kwargs)

        return wrapper

    return decorator


class RoleCheckMiddleware(BaseMiddleware):
    """Block unauthorized command execution before handlers are called."""

    async def __call__(self, handler: Callable, event: Message, data: dict) -> Any:
        if not isinstance(event, Message) or not event.text:
            return await handler(event, data)

        command = _extract_command(event)
        required = COMMAND_ROLE_REQUIREMENTS.get(command)
        if required is None:
            return await handler(event, data)

        user_id = event.from_user.id if event.from_user else 0
        if has_required_role(user_id, required):
            return await handler(event, data)

        await event.answer(
            f"Команда `{command}` недоступна для твоей роли `{get_role(user_id)}`. "
            f"Требуется `{required}`."
        )
        return None
