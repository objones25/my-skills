import httpx

from aioutil import RetryPolicy, build_client, request_with_retry


async def charge(card_token: str, cents: int) -> dict:
    async with build_client(base_url="https://payments.internal") as client:
        r = await request_with_retry(
            client, "POST", "/v1/charges",
            json={"card": card_token, "amount": cents},
            policy=RetryPolicy(attempts=4),
        )
        r.raise_for_status()
        return r.json()
