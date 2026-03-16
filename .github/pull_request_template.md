## Summary

- What problem does this PR solve?
- What is the core change?

## Scope

- [ ] `src/keil2cmake`
- [ ] `src/openocd_mcp`
- [ ] `src/k2c_tinyml`
- [ ] docs / workflow / memory only

## Behavior Changes

- User-visible CLI/API changes:
- Config / preset / template changes:
- Backward compatibility notes:

## Validation

List only commands you actually ran and the result.

```bash
# keil2cmake
uv run --with jinja2 python -m unittest tests.test_sync_keil tests.test_packaging_spec tests.test_i18n_and_generation tests.test_extra_coverage.TestUvprojxAndCli -v

# openocd_mcp
cd src/openocd_mcp
uv run --with fastmcp --with pyserial --with pyelftools --with cmsis-svd --with pyusb --with pytest python -m pytest -q tests/openocd_mcp
```

## Risks and Rollback

- Main risks:
- Rollback plan:

## Review Checklist

- [ ] I checked for unintended file moves/deletes.
- [ ] I kept module boundaries clear (`keil2cmake` vs `openocd_mcp`).
- [ ] I updated docs for any CLI/interface change.
- [ ] I added/updated tests for behavior changes.
- [ ] I included concrete validation output in this PR.

## Codex Review

- Auto review is enabled by default.
- Mention `@codex` in this PR to manually request a focused review/task.
