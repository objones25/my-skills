---
paths:
  - "internal/store/**/*.go"
---

Wrap every store error with fmt.Errorf("store: %w", err).
