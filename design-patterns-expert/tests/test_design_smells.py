"""Every finding kind has a positive case; every defect found in the audit has a regression test."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import design_smells as ds


def kinds(path: Path, root: Path | None = None, **thresholds: int) -> list[str]:
    target = str(root if root is not None else path)
    return [f.kind for f in ds.scan([target], thresholds).findings
            if root is None or f.file == str(path)]


# ----------------------------------------------------------------- positives

POSITIVES = {
    "dispatch_chain": """
        def export(kind, data):
            if kind == "pdf": return 1
            elif kind == "html": return 2
            elif kind == "csv": return 3
        """,
    "isinstance_chain": """
        def render(node):
            if isinstance(node, int): return "i"
            elif isinstance(node, str): return "s"
        """,
    "long_param_list": "def f(a, b, c, d, e, g): ...\n",
    "boolean_positional": "def render(text: str, bold: bool = False): ...\n",
    "boolean_flags": "def render(text, *, bold: bool = False, italic: bool = False, caps: bool = False): ...\n",
    "long_function": "def f():\n" + "    x = 1\n" * 70,
    "deep_nesting": """
        def f(rows):
            for r in rows:
                if r:
                    for c in r:
                        while c:
                            c -= 1
        """,
    "god_class": "class Big:\n" + "".join(f"    def m{i}(self): ...\n" for i in range(11)),
    "concrete_construction": """
        class SESEmailer: ...
        def confirm(order_id):
            SESEmailer().send(order_id)
        """,
    "import_time_io": """
        import requests
        CONFIG = requests.get("https://example.com/config").json()
        """,
    "cache_on_method": """
        from functools import cache
        class Prices:
            @cache
            def lookup(self, sku): ...
        """,
}


@pytest.mark.parametrize("kind", sorted(POSITIVES))
def test_each_kind_has_a_positive(kind: str, write) -> None:
    assert kind in kinds(write("mod.py", POSITIVES[kind]))


def test_singleton_plain_assign(write) -> None:
    src = """
        class Config:
            _instance = None
            def __new__(cls):
                return cls._instance
        """
    assert "singleton" in kinds(write("m.py", src))


def test_infra_in_domain_inside_a_domain_package(write, tmp_path: Path) -> None:
    write("shop/__init__.py")
    write("shop/domain/__init__.py")
    mod = write("shop/domain/orders.py", "import sqlalchemy\n")
    assert kinds(mod, tmp_path) == ["infra_in_domain"]


def test_reversed_comparison_is_the_same_dispatch(write) -> None:
    src = """
        def f(kind):
            if "a" == kind: return 1
            elif "b" == kind: return 2
            elif "c" == kind: return 3
        """
    assert "dispatch_chain" in kinds(write("m.py", src))


# ------------------------------------------------- regressions from the audit

def test_flat_elif_chain_is_not_deep_nesting(write) -> None:
    src = """
        def classify(n):
            if n < 0: return "neg"
            elif n == 0: return "zero"
            elif n < 10: return "small"
            elif n < 100: return "medium"
            else: return "large"
        """
    assert "deep_nesting" not in kinds(write("m.py", src))


def test_match_dispatch_is_detected(write) -> None:
    src = """
        def export(kind):
            match kind:
                case "pdf": return 1
                case "html": return 2
                case "csv": return 3
                case _: raise ValueError(kind)
        """
    assert kinds(write("m.py", src)) == ["dispatch_chain"]


def test_singleton_with_annotated_classvar(write) -> None:
    src = """
        from typing import ClassVar
        class Config:
            _instance: ClassVar["Config | None"] = None
            def __new__(cls):
                return cls._instance
        """
    assert "singleton" in kinds(write("m.py", src))


def test_singleton_metaclass_by_attribute(write) -> None:
    src = """
        import meta
        class Config(metaclass=meta.SingletonMeta): ...
        """
    assert "singleton" in kinds(write("m.py", src))


@pytest.mark.parametrize("dirname", ["scorecard", "models-research", "hardcore"])
def test_domain_hint_is_a_whole_token_not_a_substring(dirname: str, write, tmp_path: Path) -> None:
    mod = write(f"{dirname}/client.py", "import requests\n")
    assert "infra_in_domain" not in kinds(mod, tmp_path / dirname)


def test_parent_dirs_outside_the_package_do_not_count(write, tmp_path: Path) -> None:
    write("core-api/app/__init__.py")
    mod = write("core-api/app/http.py", "import requests\n")
    assert "infra_in_domain" not in kinds(mod, tmp_path / "core-api")


def test_two_keyword_only_flags_are_the_recommended_fix_not_a_smell(write) -> None:
    src = "def render(text, *, bold: bool = False, italic: bool = False): ...\n"
    assert kinds(write("m.py", src)) == []


def test_property_setter_and_dunder_bools_are_exempt(write) -> None:
    src = """
        class W:
            @property
            def visible(self): return True
            @visible.setter
            def visible(self, value: bool): ...
            def __exit__(self, a, b, c, suppress: bool = False): ...
        """
    assert "boolean_positional" not in kinds(write("m.py", src))


def test_construction_in_the_composition_root_is_fine(write) -> None:
    src = """
        class SESEmailer: ...
        def main():
            SESEmailer()
        def create_app():
            SESEmailer()
        def build_emailer():
            return SESEmailer()
        """
    assert "concrete_construction" not in kinds(write("m.py", src))


def test_construction_in_tests_is_fine(write) -> None:
    src = """
        class FakeRepository: ...
        def test_it():
            FakeRepository()
        """
    assert "concrete_construction" not in kinds(write("test_orders.py", src))


def test_inner_function_of_a_factory_is_fine(write) -> None:
    src = """
        import smtplib
        def build_smtp_sender(host):
            def send(recipient, body):
                smtplib.SMTP(host).sendmail("a", [recipient], body)
            return send
        """
    assert "concrete_construction" not in kinds(write("m.py", src))


def test_adapter_packages_may_construct_infrastructure(write, tmp_path: Path) -> None:
    write("shop/__init__.py")
    write("shop/adapters/__init__.py")
    mod = write("shop/adapters/mail.py", "import smtplib\ndef send(r, b):\n    smtplib.SMTP('h').sendmail('a', [r], b)\n")
    assert "concrete_construction" not in kinds(mod, tmp_path)


def test_nested_construction_is_reported_once_on_the_inner_function(write) -> None:
    src = """
        class SESEmailer: ...
        def outer():
            def inner():
                SESEmailer()
            return inner
        """
    found = [f for f in ds.scan([str(write("m.py", src))]).findings if f.kind == "concrete_construction"]
    assert [f.detail.split("()")[0] for f in found] == ["inner"]


def test_infra_factory_via_module_is_construction(write) -> None:
    src = """
        import boto3
        def upload(data):
            boto3.client("s3").put_object(Body=data)
        """
    assert "concrete_construction" in kinds(write("m.py", src))


def test_guarded_main_block_is_not_import_time_io(write) -> None:
    src = """
        if __name__ == "__main__":
            print(open("x").read())
        """
    assert kinds(write("m.py", src)) == []


def test_cached_property_and_staticmethod_cache_are_fine(write) -> None:
    src = """
        from functools import cache, cached_property
        class Prices:
            @cached_property
            def table(self): ...
            @staticmethod
            @cache
            def lookup(sku): ...
        """
    assert kinds(write("m.py", src)) == []


def test_nested_function_with_cache_inside_a_method_is_not_a_method(write) -> None:
    src = """
        from functools import cache
        class Prices:
            def run(self):
                @cache
                def helper(self_like): ...
                return helper
        """
    assert "cache_on_method" not in kinds(write("m.py", src))


# ----------------------------------------------------------------------- CLI

def test_unknown_only_kind_is_an_error_not_a_clean_report(write, capsys) -> None:
    path = write("m.py", POSITIVES["dispatch_chain"])
    assert ds.main([str(path), "--only", "dispach_chain"]) == 2
    assert "unknown kind" in capsys.readouterr().err


def test_filtered_empty_report_does_not_claim_no_pattern(write, capsys) -> None:
    path = write("m.py", POSITIVES["dispatch_chain"])
    assert ds.main([str(path), "--only", "god_class"]) == 0
    out = capsys.readouterr().out
    assert "recommend no pattern" not in out and "Other kinds were not checked" in out


def test_fail_on_findings_exit_code(write) -> None:
    path = write("m.py", POSITIVES["dispatch_chain"])
    assert ds.main([str(path)]) == 0
    assert ds.main([str(path), "--fail-on-findings"]) == 1


def test_threshold_override_and_validation(write) -> None:
    path = write("m.py", "def f(a, b, c, d, e, g): ...\n")
    assert kinds(path, max_params=6) == []
    with pytest.raises(SystemExit) as exc:
        ds.main([str(path), "--threshold", "max_parms=3"])
    assert exc.value.code == 2


def test_json_output_includes_thresholds(write, capsys) -> None:
    path = write("m.py", POSITIVES["long_param_list"])
    ds.main([str(path), "--json", "--max-params", "4"])
    data = json.loads(capsys.readouterr().out)
    assert data["thresholds"]["max_params"] == 4
    assert data["findings"][0]["kind"] == "long_param_list"


def test_list_kinds_needs_no_paths_but_scanning_does(capsys) -> None:
    assert ds.main(["--list-kinds"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == len(ds.KINDS)
    with pytest.raises(SystemExit) as exc:
        ds.main([])
    assert exc.value.code == 2


def test_every_kind_is_documented() -> None:
    covered = set(POSITIVES) | {"singleton", "infra_in_domain"}
    assert covered == set(ds.KINDS)
