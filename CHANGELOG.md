# Changelog

## Unreleased

### Fixed

- Silence successful managed-command completions from native delegation children before they can wake the parent model.
- Fixed cache status display so missing telemetry is not shown as a miss and positive sub-1% cache hits are not rendered as `0%`.
