import asyncio


async def send_notification(user_id: int) -> None:
    await asyncio.sleep(0)


async def handle_signup(user_id: int) -> str:
    send_notification(user_id)
    return "ok"


def test_signup():
    assert asyncio.run(handle_signup(1)) == "ok"
