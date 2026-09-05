from __future__ import annotations

import httpx

from local_agents.security.workspace import Workspace
from local_agents.tools.forms import FormFieldsTool, parse_form_fields
from local_agents.tools.web import make_http_client

FORM = """<html><body><script>var x = "<input name=fake>";</script>
<h1>Apply</h1>
<form>
 <label for="fn">First name *</label><input id="fn" name="first_name" type="text" required>
 <label>Email <input type="email" name="email" placeholder="you@example.com"></label>
 <p>Years of experience</p>
 <select name="years"><option value="">Select...</option><option>0-2</option><option>3-5</option><option>6+</option></select>
 <fieldset><legend>Work authorization</legend>
   <label><input type="radio" name="auth" value="yes"> Yes</label>
   <label><input type="radio" name="auth" value="no"> No</label>
 </fieldset>
 <input type="checkbox" id="remote" name="prefs" value="remote"><label for="remote">Open to remote</label>
 <input type="checkbox" id="relocate" name="prefs" value="relocate"><label for="relocate">Willing to relocate</label>
 <label for="why">Why do you want to work here?</label><textarea id="why" name="why" required></textarea>
 <input type="hidden" name="csrf" value="x"><input type="submit" value="Send">
</form>
<form><input name="q" placeholder="Search"></form>
</body></html>"""


def test_parse_form_fields() -> None:
    forms = parse_form_fields(FORM)
    assert len(forms) == 2
    fields = {f.name: f for f in forms[0]}
    assert list(fields) == ["first_name", "email", "years", "auth", "prefs", "why"]
    assert fields["first_name"].label == "First name *" and fields["first_name"].required
    assert fields["email"].kind == "email" and fields["email"].label == "Email"
    assert fields["email"].placeholder == "you@example.com"
    assert fields["years"].options == ["0-2", "3-5", "6+"]
    assert fields["years"].label == "Years of experience"
    assert fields["auth"].kind == "radio" and fields["auth"].options == ["Yes", "No"]
    assert fields["auth"].label == "Work authorization"
    assert fields["prefs"].options == ["Open to remote", "Willing to relocate"]
    assert fields["why"].kind == "textarea" and fields["why"].required
    assert "csrf" not in fields and "fake" not in fields
    assert forms[1][0].name == "q"
    assert "options: 0-2 | 3-5 | 6+" in fields["years"].describe()


def test_no_forms() -> None:
    assert parse_form_fields("<p>nothing here</p>") == []


async def test_tool_from_url_and_path(workspace: Workspace) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/js":
            return httpx.Response(200, text="<div id=app></div>")
        return httpx.Response(200, text=FORM)

    tool = FormFieldsTool(workspace, make_http_client(transport=httpx.MockTransport(handler)))
    result = await tool.run(url="https://jobs.test/apply")
    assert result.success and "7 field(s) in 2 form(s)" in result.output
    assert "Why do you want to work here?" in result.output
    assert result.data[0][0]["name"] == "first_name"
    empty = await tool.run(url="https://jobs.test/js")
    assert empty.success and "No form fields found" in empty.output
    (workspace.root / "survey.html").write_text(FORM)
    local = await tool.run(path="survey.html")
    assert (local.success and "survey.html" in result.output) or local.success
    neither = await tool.run()
    assert not neither.success
    outside = await tool.run(path="../x.html")
    assert outside.blocked
