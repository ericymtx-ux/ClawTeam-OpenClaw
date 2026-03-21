"""Tests for clawteam.templates — loading, parsing, and variable substitution."""

import pytest
from pydantic import ValidationError

from clawteam.templates import (
    AgentDef,
    TaskDef,
    TemplateDef,
    _SafeDict,
    list_templates,
    load_template,
    render_task,
)


class TestRenderTask:
    def test_basic_substitution(self):
        result = render_task("Analyze {goal} for {team_name}", goal="AAPL", team_name="alpha")
        assert result == "Analyze AAPL for alpha"

    def test_unknown_placeholders_kept(self):
        """Variables we don't provide should stay as {placeholder}."""
        result = render_task("Hello {name}, team is {team_name}", name="bob")
        assert result == "Hello bob, team is {team_name}"

    def test_no_variables(self):
        result = render_task("plain text with no placeholders")
        assert result == "plain text with no placeholders"

    def test_empty_string(self):
        assert render_task("") == ""

    def test_multiple_same_variable(self):
        result = render_task("{x} and {x}", x="foo")
        assert result == "foo and foo"


class TestSafeDict:
    def test_missing_key_returns_placeholder(self):
        d = _SafeDict(a="1")
        assert d["a"] == "1"
        # missing key wrapped back into braces
        assert "{missing}".format_map(d) == "{missing}"


class TestModels:
    def test_agent_def_defaults(self):
        a = AgentDef(name="worker")
        assert a.type == "general-purpose"
        assert a.task == ""
        assert a.command is None

    def test_task_def(self):
        t = TaskDef(subject="Build feature", description="details", owner="alice")
        assert t.subject == "Build feature"

    def test_template_def_defaults(self):
        leader = AgentDef(name="lead")
        t = TemplateDef(name="my-tmpl", leader=leader)
        assert t.description == ""
        assert t.command == ["openclaw"]
        assert t.backend == "tmux"
        assert t.agents == []
        assert t.tasks == []


class TestModelFields:
    def test_agent_def_model_defaults_none(self):
        a = AgentDef(name="worker")
        assert a.model is None
        assert a.model_tier is None

    def test_agent_def_with_model(self):
        a = AgentDef(name="worker", model="opus")
        assert a.model == "opus"

    def test_agent_def_with_valid_tier(self):
        a = AgentDef(name="worker", model_tier="strong")
        assert a.model_tier == "strong"

    def test_agent_def_invalid_tier_raises(self):
        with pytest.raises(ValidationError, match="model_tier"):
            AgentDef(name="worker", model_tier="ultra")

    def test_template_def_model_defaults(self):
        leader = AgentDef(name="lead")
        t = TemplateDef(name="test", leader=leader)
        assert t.model is None
        assert t.model_strategy is None

    def test_template_def_with_model(self):
        leader = AgentDef(name="lead")
        t = TemplateDef(name="test", leader=leader, model="sonnet-4.6")
        assert t.model == "sonnet-4.6"

    def test_template_def_valid_strategy(self):
        leader = AgentDef(name="lead")
        t = TemplateDef(name="test", leader=leader, model_strategy="auto")
        assert t.model_strategy == "auto"

    def test_template_def_none_strategy(self):
        leader = AgentDef(name="lead")
        t = TemplateDef(name="test", leader=leader, model_strategy="none")
        assert t.model_strategy == "none"

    def test_template_def_invalid_strategy_raises(self):
        leader = AgentDef(name="lead")
        with pytest.raises(ValidationError, match="model_strategy"):
            TemplateDef(name="test", leader=leader, model_strategy="magic")


class TestParseTomlWithModel:
    def test_toml_with_agent_model(self, tmp_path, monkeypatch):
        toml_content = """\
[template]
name = "model-test"
model = "sonnet-4.6"

[template.leader]
name = "boss"
model = "opus"

[[template.agents]]
name = "worker"
model_tier = "cheap"
"""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "model-test.toml").write_text(toml_content)

        import clawteam.templates as tmod
        monkeypatch.setattr(tmod, "_USER_DIR", tpl_dir)

        tmpl = load_template("model-test")
        assert tmpl.model == "sonnet-4.6"
        assert tmpl.leader.model == "opus"
        assert tmpl.agents[0].model_tier == "cheap"

    def test_toml_with_strategy(self, tmp_path, monkeypatch):
        toml_content = """\
[template]
name = "strategy-test"
model_strategy = "auto"

[template.leader]
name = "boss"
type = "leader"
"""
        tpl_dir = tmp_path / "templates"
        tpl_dir.mkdir()
        (tpl_dir / "strategy-test.toml").write_text(toml_content)

        import clawteam.templates as tmod
        monkeypatch.setattr(tmod, "_USER_DIR", tpl_dir)

        tmpl = load_template("strategy-test")
        assert tmpl.model_strategy == "auto"


class TestLoadBuiltinTemplate:
    def test_load_hedge_fund(self):
        tmpl = load_template("hedge-fund")
        assert tmpl.name == "hedge-fund"
        assert tmpl.leader.name == "portfolio-manager"
        assert len(tmpl.agents) > 0
        assert len(tmpl.tasks) > 0

    def test_leader_type(self):
        tmpl = load_template("hedge-fund")
        assert tmpl.leader.type == "portfolio-manager"

    def test_agents_have_tasks(self):
        tmpl = load_template("hedge-fund")
        for agent in tmpl.agents:
            assert agent.task != "", f"Agent '{agent.name}' has no task text"

    def test_task_owners_match_agents(self):
        tmpl = load_template("hedge-fund")
        agent_names = {tmpl.leader.name} | {a.name for a in tmpl.agents}
        for task in tmpl.tasks:
            if task.owner:
                assert task.owner in agent_names, f"Task owner '{task.owner}' not in agents"


class TestLoadTemplateNotFound:
    def test_missing_template_raises(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_template("this-does-not-exist-anywhere")


class TestUserTemplateOverride:
    def test_user_template_takes_priority(self, tmp_path, monkeypatch):
        """User templates in ~/.clawteam/templates/ override builtins."""
        user_tpl_dir = tmp_path / ".clawteam" / "templates"
        user_tpl_dir.mkdir(parents=True)

        toml_content = """\
[template]
name = "custom"
description = "User override"

[template.leader]
name = "my-leader"
type = "custom-leader"
"""
        (user_tpl_dir / "custom.toml").write_text(toml_content)

        # patch the module-level _USER_DIR
        import clawteam.templates as tmod

        monkeypatch.setattr(tmod, "_USER_DIR", user_tpl_dir)

        tmpl = load_template("custom")
        assert tmpl.name == "custom"
        assert tmpl.leader.name == "my-leader"
        assert tmpl.description == "User override"


class TestListTemplates:
    def test_list_includes_builtin(self):
        templates = list_templates()
        names = {t["name"] for t in templates}
        assert "hedge-fund" in names

    def test_list_entry_format(self):
        templates = list_templates()
        for t in templates:
            assert "name" in t
            assert "description" in t
            assert "source" in t
            assert t["source"] in ("builtin", "user")
