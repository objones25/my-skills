"""Stateful property test: random sequences of operations checked against a ledger model."""

from decimal import Decimal

from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, initialize, invariant, rule
from wallet.core import InsufficientFunds, Wallet, transfer

money = st.decimals(min_value=Decimal("0.01"), max_value=Decimal("1000.00"), places=2)


class WalletMachine(RuleBasedStateMachine):
    wallets = Bundle("wallets")

    @initialize()
    def start_ledger(self):
        self.deposited = Decimal("0.00")
        self.withdrawn = Decimal("0.00")
        self.all_wallets: list[Wallet] = []

    @rule(target=wallets, owner=st.from_regex(r"[a-z]{1,8}", fullmatch=True))
    def open_wallet(self, owner):
        wallet = Wallet(owner=owner)
        self.all_wallets.append(wallet)
        return wallet

    @rule(wallet=wallets, amount=money)
    def deposit(self, wallet, amount):
        wallet.deposit(amount)
        self.deposited += amount

    @rule(wallet=wallets, amount=money)
    def withdraw(self, wallet, amount):
        try:
            wallet.withdraw(amount)
        except InsufficientFunds:
            return
        self.withdrawn += amount

    @rule(source=wallets, dest=wallets, amount=money)
    def move(self, source, dest, amount):
        try:
            transfer(source, dest, amount)
        except InsufficientFunds:
            return

    @invariant()
    def money_is_conserved(self):
        assert sum(w.balance for w in self.all_wallets) == self.deposited - self.withdrawn

    @invariant()
    def no_balance_is_negative(self):
        assert all(w.balance >= 0 for w in self.all_wallets)


TestWalletMachine = WalletMachine.TestCase
