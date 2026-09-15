# Legacy single-config entry point. Prefer configs/train_flags/*.sh.
# Hyperparams aligned with gputnam training-SBND/iterE (+ wall-time fixes).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_FLAGS="$(cd "${SCRIPT_DIR}/../../configs/train_flags" && pwd)"
# shellcheck source=../../configs/train_flags/linear.sh
source "${REPO_FLAGS}/linear.sh"
