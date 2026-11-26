"""check_setup.py lints the project configuration the setup skill writes."""
from __future__ import annotations

import json

from conftest import write, write_json


def _codes(result):
    return {line.split()[1] for line in result.stdout.splitlines() if line.startswith(("ERROR", "WARN"))}


def test_clean_project_passes(clean_project, run_script):
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 0, result.stdout
    assert _codes(result) == set()


def test_not_a_directory_cannot_be_checked(tmp_path, run_script):
    result = run_script("check_setup.py", str(tmp_path / "missing"), cwd=tmp_path)
    assert result.returncode == 2


def test_invalid_settings_json_is_an_error(clean_project, run_script):
    write(clean_project / ".claude" / "settings.json", "{\"permissions\": {,}")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST001" in _codes(result)


def test_permission_rule_that_is_not_a_string_is_an_error(clean_project, run_script):
    write_json(clean_project / ".claude" / "settings.json", {"permissions": {"allow": [{"Bash": "ls"}]}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST001" in _codes(result)


def test_local_settings_not_gitignored_is_an_error(clean_project, run_script):
    write(clean_project / ".gitignore", ".env\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST002" in _codes(result)
    assert "settings.local.json" in result.stdout


def test_claude_local_md_not_gitignored_is_an_error(clean_project, run_script):
    write(clean_project / ".gitignore", ".claude/settings.local.json\n")
    write(clean_project / "CLAUDE.local.md", "my notes\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST002" in _codes(result)
    assert "CLAUDE.local.md" in result.stdout


def test_gitignore_is_not_checked_outside_a_git_repo(clean_project, run_script):
    (clean_project / ".gitignore").unlink()
    import shutil
    shutil.rmtree(clean_project / ".git")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST002" not in _codes(result)
    assert result.returncode == 0


def test_unresolved_import_is_an_error(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n\n@docs/missing.md\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST003" in _codes(result)
    assert "docs/missing.md" in result.stdout


def test_import_inside_code_is_ignored(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n\nMention `@scope/pkg` here.\n\n```\n@not/a/file\n```\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST003" not in _codes(result)


def test_email_address_is_not_an_import(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n\nContact dev@example.com.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST003" not in _codes(result)


def test_long_claude_md_warns_without_failing(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n" + "- a rule\n" * 250)
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 0
    assert "ST004" in _codes(result)
    assert "200" in result.stdout


def test_long_claude_md_fails_under_strict(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n" + "- a rule\n" * 250)
    result = run_script("check_setup.py", "--strict", cwd=clean_project)
    assert result.returncode == 1


def test_claude_md_at_200_lines_does_not_warn(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "- a rule\n" * 200)
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST004" not in _codes(result)


def test_missing_claude_md_warns(clean_project, run_script):
    (clean_project / "CLAUDE.md").unlink()
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST010" in _codes(result)


def test_claude_md_in_dot_claude_counts(clean_project, run_script):
    text = (clean_project / "CLAUDE.md").read_text()
    (clean_project / "CLAUDE.md").unlink()
    write(clean_project / ".claude" / "CLAUDE.md", text.replace("@docs/arch.md", "@../docs/arch.md"))
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST010" not in _codes(result)
    assert "ST003" not in _codes(result)


def test_agent_without_description_is_an_error(clean_project, run_script):
    write(clean_project / ".claude" / "agents" / "bad.md", "---\nname: bad\n---\n\nBody.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST005" in _codes(result)


def test_agent_without_frontmatter_is_an_error(clean_project, run_script):
    write(clean_project / ".claude" / "agents" / "bare.md", "Just a prompt.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST005" in _codes(result)


def test_skill_name_may_differ_from_folder(clean_project, run_script):
    # skills docs: `.claude/skills/deploy-staging/` with `name: deploy` runs as /deploy
    write(clean_project / ".claude" / "skills" / "deploy-staging" / "SKILL.md",
          "---\nname: deploy\ndescription: Deploy. Use when deploying.\n---\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 0
    assert "ST006" not in _codes(result)


def test_skill_without_description_warns_without_failing(clean_project, run_script):
    write(clean_project / ".claude" / "skills" / "deploy" / "SKILL.md", "---\nname: deploy\n---\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 0
    assert "ST006" in _codes(result)


def test_literal_secret_in_mcp_env_is_an_error(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"mcpServers": {"gh": {
        "command": "npx", "env": {"GITHUB_TOKEN": "ghp_abcdefghijklmnopqrstuvwxyz0123456789"}}}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST007" in _codes(result)
    assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in result.stdout


def test_literal_secret_in_mcp_header_is_an_error(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"mcpServers": {"api": {
        "type": "http", "url": "https://example.com/mcp",
        "headers": {"Authorization": "Bearer sk-live-1234567890abcdef"}}}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" in _codes(result)
    assert "sk-live-1234567890abcdef" not in result.stdout


def test_env_reference_with_default_is_not_a_secret(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"mcpServers": {"api": {
        "type": "http", "url": "${API_BASE_URL:-https://example.com}/mcp",
        "headers": {"Authorization": "Bearer ${API_KEY}"}}}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" not in _codes(result)


def test_literal_secret_in_committed_settings_env_is_an_error(clean_project, run_script):
    settings = json.loads((clean_project / ".claude" / "settings.json").read_text())
    settings["env"]["OPENAI_API_KEY"] = "sk-proj-abcdef1234567890"
    write_json(clean_project / ".claude" / "settings.json", settings)
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" in _codes(result)
    assert "sk-proj-abcdef1234567890" not in result.stdout


def test_non_secret_env_value_is_fine(clean_project, run_script):
    settings = json.loads((clean_project / ".claude" / "settings.json").read_text())
    settings["env"]["MAX_TOKENS_PER_REQUEST"] = "4096"
    write_json(clean_project / ".claude" / "settings.json", settings)
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" not in _codes(result)


def test_mcp_json_without_servers_object_is_an_error(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"servers": {}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 1
    assert "ST008" in _codes(result)


def test_rule_paths_must_be_a_list_or_string(clean_project, run_script):
    write(clean_project / ".claude" / "rules" / "bad.md", "---\npaths:\n---\n\nRule.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST009" in _codes(result)


def test_rule_with_unknown_frontmatter_key_warns(clean_project, run_script):
    write(clean_project / ".claude" / "rules" / "odd.md", "---\nglobs: \"*.py\"\n---\n\nRule.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST009" in _codes(result)
    assert "paths" in result.stdout


def test_json_output_lists_findings(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "@nope.md\n")
    result = run_script("check_setup.py", "--json", cwd=clean_project)
    data = json.loads(result.stdout)
    assert [f["code"] for f in data["findings"]] == ["ST003"]
    assert data["findings"][0]["severity"] == "error"


def test_block_scalar_skill_description_is_accepted(clean_project, run_script):
    write(clean_project / ".claude" / "skills" / "deploy" / "SKILL.md",
          "---\nname: deploy\ndescription: >-\n  Deploy the app.\n  Use when deploying.\n---\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST006" not in _codes(result), result.stdout


def test_empty_block_scalar_description_is_missing(clean_project, run_script):
    write(clean_project / ".claude" / "skills" / "deploy" / "SKILL.md",
          "---\nname: deploy\ndescription: >-\n---\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST006" in _codes(result)


def test_non_utf8_instruction_file_cannot_be_checked(clean_project, run_script):
    (clean_project / "CLAUDE.md").write_bytes(b"# Demo\n\xff\xfe\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "CLAUDE.md" in result.stderr


def test_import_followed_by_sentence_punctuation_resolves(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n\nSee @docs/arch.md. Also (@docs/arch.md), fine.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST003" not in _codes(result), result.stdout


def test_import_of_a_directory_is_an_error(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n\n@docs/\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST003" in _codes(result)


def test_nested_code_fence_hides_its_imports(clean_project, run_script):
    write(clean_project / "CLAUDE.md", "# Demo\n\n````md\n```\n@inner/missing.md\n```\n@still/inside.md\n````\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST003" not in _codes(result), result.stdout


def test_secret_in_mcp_args_is_an_error(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"mcpServers": {"gh": {
        "command": "npx", "args": ["-y", "srv", "--token=ghp_abcdefghijklmnop1234"]}}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" in _codes(result)
    assert "ghp_abcdefghijklmnop1234" not in result.stdout


def test_secret_in_mcp_url_query_is_an_error(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"mcpServers": {"api": {
        "type": "http", "url": "https://example.com/mcp?api_key=abcdefghijklmnop1234"}}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" in _codes(result)
    assert "abcdefghijklmnop1234" not in result.stdout


def test_url_query_with_env_reference_is_fine(clean_project, run_script):
    write_json(clean_project / ".mcp.json", {"mcpServers": {"api": {
        "type": "http", "url": "https://example.com/mcp?api_key=${API_KEY}"}}})
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST007" not in _codes(result)


def test_flow_style_empty_paths_is_an_error(clean_project, run_script):
    write(clean_project / ".claude" / "rules" / "bad.md", "---\npaths: []\n---\n\nRule.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST009" in _codes(result)


def test_flow_style_paths_list_is_fine(clean_project, run_script):
    write(clean_project / ".claude" / "rules" / "ok.md", "---\npaths: [\"src/**/*.py\", \"tests/**\"]\n---\n\nRule.\n")
    result = run_script("check_setup.py", cwd=clean_project)
    assert "ST009" not in _codes(result)
