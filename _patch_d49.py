import pathlib

p = pathlib.Path("app/tests/test_phase10_recorded_gate.py")
s = p.read_text(encoding="utf-8")

# Cut the mangled block between the two marker lines and rewrite it cleanly.
start_marker = "    calls = []\n\n    def fake_run(argv, **kwargs):\n        calls.append(list(argv))\n        if argv[:4] == [\"docker\", \"compose\", \"images\", \"-q\"]:"
end_marker = "    # Happy path: current containers still win."

start = s.index(start_marker)
end = s.index(end_marker)

tab = chr(92) + "t"   # literal backslash-t for the file's string literal
nl = chr(92) + "n"    # literal backslash-n

block = (
    "    calls = []\n"
    "\n"
    "    # REAL compose config shape for this project: build-only services carry\n"
    "    # no `image` key (config emits None), so the fallback must go through\n"
    "    # the compose project/service labels on the freshly built image.\n"
    "    real_config_shape = {\n"
    "        \"name\": \"rag-vector-database-pipeline-project\",\n"
    "        \"services\": {\n"
    "            \"api\": {\n"
    "                \"build\": {\"context\": \".\", \"dockerfile\": \"Dockerfile\"},\n"
    "                \"image\": None,\n"
    "            }\n"
    "        },\n"
    "    }\n"
    "\n"
    "    def fake_run(argv, **kwargs):\n"
    "        calls.append(list(argv))\n"
    "        if argv[:4] == [\"docker\", \"compose\", \"images\", \"-q\"]:\n"
    "            return mock.Mock(returncode=1, stdout=\"\", stderr=\"No such image: sha256:dead\")\n"
    "        if argv[:3] == [\"docker\", \"compose\", \"config\"]:\n"
    "            return mock.Mock(returncode=0, stdout=_json.dumps(real_config_shape), stderr=\"\")\n"
    "        if argv[:3] == [\"docker\", \"image\", \"ls\"]:\n"
    "            listing = \"sha256:staleold" + tab + "2026-08-14 09:00:00" + nl \
    + "sha256:freshlybuilt" + tab + "2026-08-15 15:18:09" + nl + "\"\n"
    "            return mock.Mock(returncode=0, stdout=listing, stderr=\"\")\n"
    "        raise AssertionError(f\"unexpected argv {argv}\")\n"
    "\n"
    "    monkeypatch.setattr(binding.subprocess, \"run\", fake_run)\n"
    "    assert binding._resolve_image_id(\"api\") == \"sha256:freshlybuilt\"\n"
    "    assert any(\n"
    "        \"--filter\" in argv and \"com.docker.compose.project=\" in \" \".join(argv)\n"
    "        for argv in calls\n"
    "    )\n"
    "\n"
)

s = s[:start] + block + s[end:]
p.write_text(s, encoding="utf-8")
print("patched")
