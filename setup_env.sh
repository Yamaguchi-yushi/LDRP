#!/usr/bin/env bash
#
# LDRP conda env セットアップスクリプト.
#
# 使い方:
#   ./setup_env.sh                    # ldrp env がなければ作成 + 依存をインストール
#   ./setup_env.sh --recreate         # 既存の ldrp env を削除してゼロから作り直す
#   LDRP_ENV_NAME=foo ./setup_env.sh  # env 名を foo にする
#
# 前提:
#   - anaconda / miniconda がインストール済みで `conda` コマンドが使える
#   - リポジトリのルートディレクトリで実行する (= setup_env.sh と同じディレクトリ)
#
set -euo pipefail

ENV_NAME="${LDRP_ENV_NAME:-ldrp}"
PYTHON_VERSION="3.9"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RECREATE=false
for arg in "$@"; do
    case "$arg" in
        --recreate) RECREATE=true ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "[setup] 不明な引数: $arg (use --help)" >&2
            exit 2
            ;;
    esac
done

# --- 1) conda の存在確認 -------------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] ERROR: conda が見つかりません. anaconda / miniconda をインストールしてください." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
ENV_PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"

# --- 2) 既存 env のハンドリング ------------------------------------------------
env_exists() {
    conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$1"
}

if env_exists "$ENV_NAME"; then
    if [ "$RECREATE" = "true" ]; then
        echo "[setup] 既存 env '${ENV_NAME}' を削除します..."
        conda env remove -n "$ENV_NAME" -y
    else
        echo "[setup] env '${ENV_NAME}' は既に存在します. (作り直したい場合は --recreate)"
    fi
fi

# --- 3) env 作成 ----------------------------------------------------------------
if ! env_exists "$ENV_NAME"; then
    echo "[setup] conda env '${ENV_NAME}' を作成中 (python ${PYTHON_VERSION})..."
    conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi

# --- 4) pip の更新 + 依存インストール -----------------------------------------
echo "[setup] pip を更新中..."
"$ENV_PYTHON" -m pip install --upgrade pip --quiet

echo "[setup] 依存パッケージをインストール中 (requirements.txt)..."
"$ENV_PYTHON" -m pip install -r "${REPO_ROOT}/requirements.txt"

# --- 5) ローカル drp パッケージを editable インストール -----------------------
echo "[setup] ローカルの drp パッケージを editable モードでインストール中..."
"$ENV_PYTHON" -m pip install -e "${REPO_ROOT}/src/main" --quiet

# --- 6) 動作確認 ----------------------------------------------------------------
echo "[setup] 動作確認中..."
"$ENV_PYTHON" - <<EOF
import gym, numpy, torch, networkx, yaml
print(f"  gym      : {gym.__version__}")
print(f"  numpy    : {numpy.__version__}")
print(f"  torch    : {torch.__version__}")
print(f"  networkx : {networkx.__version__}")
print(f"  PyYAML   : {yaml.__version__}")

import sys
sys.path.append("${REPO_ROOT}")
sys.path.append("${REPO_ROOT}/src/main")
import drp_env
print(f"  drp_env  : {drp_env.__file__}")
print("[setup] verification OK")
EOF

cat <<EOF

[setup] done.

env を activate するには:
  conda activate ${ENV_NAME}

env を activate せずに直接 python を使うには:
  ${ENV_PYTHON} test.py

env を作り直したい場合:
  ./setup_env.sh --recreate

EOF
