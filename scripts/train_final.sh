#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
Full final training uses public teachers to create soft labels, then distills
official-data student checkpoints. The packaged model is already trained.

For score reproduction, use:
  bash run_reproduce.sh /path/to/DLUT_VLG_2026_本科生/data --cuda

The original distillation training scripts used in the experiment are also
included as run_distill_val_hard.sh and run_distill_test_hard.sh.
EOF
