## Summary

<!-- What changed and why? -->

## Type of change

- [ ] Runtime mouse API/path execution
- [ ] Runtime keyboard API
- [ ] RL environment/training/policy
- [ ] Packaged model or model metadata
- [ ] Documentation
- [ ] Packaging/release automation
- [ ] Other

## Validation

Please check every command you ran:

- [ ] `uv run black --check .`
- [ ] `uv run isort --check-only .`
- [ ] `uv run python -m compileall scripts src`
- [ ] `uv build && uv run twine check dist/*`
- [ ] Packaged model asset check passed, if packaging changed
- [ ] Manual smoke test, if runtime behavior changed

## Model/release impact

- [ ] This PR does not change the packaged model.
- [ ] This PR changes `src/bumblebee/models/sac_mouse_v2.zip`.
- [ ] If the model changed, `manifest.json` was regenerated.
- [ ] If the model changed, `MODEL_CARD.md` and release notes were updated.
- [ ] No datasets, checkpoints, replay buffers, or TensorBoard logs were committed.

## Safety notes

<!-- Mention any real mouse/keyboard behavior, permissions, fail-safe behavior, or commands that move/type on the host machine. -->

## Screenshots / traces / examples

<!-- Optional: screenshots, trajectory screenshots, terminal output, or sample code. -->
