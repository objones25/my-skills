"""v0 wallet: float amounts, validation by comparison. NaN slips through both checks."""


class InsufficientFunds(ValueError):
    pass


class Wallet:
    def __init__(self, balance: float = 0.0) -> None:
        self.balance = balance

    def withdraw(self, amount: float) -> float:
        if amount <= 0:
            raise ValueError(f"withdrawal must be positive, got {amount}")
        if amount > self.balance:
            raise InsufficientFunds(f"cannot withdraw {amount} from balance {self.balance}")
        self.balance -= amount
        return self.balance
