# LDRP 詳細説明書

> **LDRP** = Learning-based Dynamic Robot/Drone Path Planning  
> 複数エージェントによる動的経路計画とタスク割り当てを扱う Multi-Agent Reinforcement Learning (MARL) フレームワーク

---

## 目次

1. [プロジェクトの概要](#1-プロジェクトの概要)
2. [MARL4DRP との違い](#2-marl4drp-との違い)
3. [ディレクトリ構成](#3-ディレクトリ構成)
4. [インストール方法](#4-インストール方法)
5. [実行方法](#5-実行方法)
6. [設定ファイルの詳細](#6-設定ファイルの詳細)
7. [アルゴリズムの解説](#7-アルゴリズムの解説)
8. [環境の仕様](#8-環境の仕様)
9. [独自 Policy の実装方法](#9-独自-policy-の実装方法)
10. [学習・モデル適用方法](#10-学習モデル適用方法)
11. [注意事項・既知の問題](#11-注意事項既知の問題)

---

## 1. プロジェクトの概要

LDRP は，複数のロボット・ドローン（エージェント）がグラフ構造のマップ上を移動しながら，配送タスク（ピックアップ → 配達）を効率的にこなすことを目的とした環境・フレームワークです．

### 解いている問題

- **DRP (Drone Routing Problem)**: 複数エージェントが衝突なく目標ノードへ経路を計画する問題
- **LDRP (Learning-based DRP)**: これに「タスク割り当て」を組み合わせ，継続的に発生する配送タスクを学習ベースで最適化する問題

### 主な特徴

| 特徴 | 内容 |
|------|------|
| **タスク割り当て機能** | ピックアップ・配達タスクを動的に割り当て |
| **複数の経路計画アルゴリズム** | PBS（記号的探索）および学習済み MARL モデル（IQL, QMIX 等）に対応 |
| **複数のタスク割り当てアルゴリズム** | FIFO, TP（最短距離ベース）, PPO（学習ベース）に対応 |
| **13 種類のマップ** | 小規模グリッドから実地図（青葉，渋谷，京大，パリ等）まで |
| **Gym 互換インターフェース** | OpenAI Gym 準拠 |

---

## 2. MARL4DRP との違い

[MARL4DRP](https://github.com/Yamaguchi-yushi/MARL4DRP) は LDRP の前身・ベースとなるプロジェクトです．両者の主な違いを以下に整理します．

### 機能比較表

| 項目 | MARL4DRP | LDRP |
|------|----------|------|
| **対象問題** | DRP（目標地点への到達） | LDRP（継続的な配送タスク） |
| **タスク割り当て** | なし（固定ゴール） | あり（動的タスク生成・割り当て） |
| **経路計画アルゴリズム** | IQL, QMIX, VDN | PBS + IQL, QMIX（拡張） |
| **タスク割り当てアルゴリズム** | — | FIFO, TP, PPO |
| **マップ数** | 8 種類（3×3 〜 10×10 + aoba） | 13 種類（実地図を追加：渋谷，京大，大阪，パリ，四条） |
| **エージェント数上限** | 最大 6 エージェント | 最大 30 エージェント |
| **衝突報酬** | −50 × speed（speed=5 → −250） | −100（固定値） |
| **Gym 環境名** | `drp-Nagent_map-v2` | `drp-Nagent_map-v2` + `drp_safe-Nagent_map-v2` |
| **セーフモード** | なし | SafeEnv ラッパーで無効行動を防止 |
| **学習フレームワーク** | epymarl v1.0.0（固定） | epymarl（サブモジュール，バージョン柔軟） |
| **バッチ実行** | なし | `run.py` で複数条件を並列実行 |
| **インタラクティブ GUI** | なし | `drpload_test.py`（matplotlib） |

### アーキテクチャの違い

```
MARL4DRP（シンプルな DRP）
  ┌──────────────┐
  │  MARL Policy  │  ← IQL / QMIX / VDN のみ
  │   (epymarl)   │
  └──────┬───────┘
         │ action（目標ノード）
  ┌──────▼───────┐
  │   DrpEnv      │  ← 固定ゴール，衝突で即終了
  └──────────────┘


LDRP（タスク付き DRP）
  ┌──────────────────────────────────┐
  │           Policy (統合)           │
  │  ┌─────────────┐ ┌─────────────┐ │
  │  │ PolicyManager│ │TaskManager  │ │
  │  │ PBS / MARL   │ │FIFO/TP/PPO │ │
  │  └──────┬──────┘ └──────┬──────┘ │
  └─────────┼───────────────┼────────┘
            │ pass          │ task
  ┌─────────▼───────────────▼────────┐
  │  joint_action = {"pass":...,      │
  │                  "task":...}      │
  └──────────────────────────────────┘
            │
  ┌─────────▼────────────────────────┐
  │  TaskEnv / SafeEnv (ラッパー)     │
  │       ↕                          │
  │        DrpEnv                    │  ← 継続タスク，衝突しても継続可能
  └──────────────────────────────────┘
```

### 問題設定の違い（重要）

| 項目 | MARL4DRP | LDRP |
|------|----------|------|
| **ゴール** | 全エージェントが各自のゴールノードに到達 | 全タスク（ピックアップ + 配達）を完了 |
| **エピソード終了条件** | 全員ゴール OR 衝突 OR 時間切れ | タスク完了数が最大化されるまで継続 OR 時間切れ |
| **タスクの流れ** | なし | タスク割り当て → ピックアップ → 輸送 → 配達 → 次のタスク待ち |
| **タスク発生** | なし | 各ステップで `current_tasklist` に動的追加 |

---

## 3. ディレクトリ構成

```
LDRP/
├── test.py               # 単一テスト実行のエントリポイント
├── run.py                # 複数条件をまとめて実験（並列実行）
├── train.py              # MARL モデルの学習
├── runner.py             # エピソード実行ループ
├── drpload_test.py       # PBS のインタラクティブ GUI テスト
│
├── src/
│   ├── config/
│   │   └── default.yaml          # 全体設定（マップ・エージェント数・アルゴリズム等）
│   │
│   ├── policy.py                 # PolicyManager + TaskManager を統合するクラス
│   │
│   ├── all_policy/               # 経路計画アルゴリズム
│   │   ├── policy_manager.py     # PBS か MARL かを選択
│   │   ├── pbs.py                # Priority-Based Search（記号的探索）
│   │   ├── policy.py             # MARL モデルの推論ラッパー
│   │   └── policy_runner.py      # モデルファイル読み込み・推論
│   │
│   ├── task_assign/              # タスク割り当てアルゴリズム
│   │   ├── task_manager.py       # アルゴリズム選択
│   │   └── task_policy/
│   │       ├── random.py         # FIFO（順番割り当て）
│   │       ├── tp.py             # Task Priority（最短距離ベースの貪欲法）
│   │       └── ppo.py            # PPO（学習ベース）
│   │
│   ├── main/                     # Gym 環境の本体
│   │   ├── setup.py              # 環境登録（gym.make に対応）
│   │   ├── requirements.txt      # 依存ライブラリ
│   │   ├── drp_env/
│   │   │   ├── drp_env.py        # メイン環境クラス（step, reset, reward）
│   │   │   ├── EE_map.py         # マップ読み込み・グラフ構築（NetworkX）
│   │   │   ├── map/              # マップデータ（CSV）
│   │   │   │   ├── map_3x3/      # 3×3 グリッド（最小）
│   │   │   │   ├── map_5x4/
│   │   │   │   ├── map_8x5/
│   │   │   │   ├── map_10x6/
│   │   │   │   ├── map_10x8/
│   │   │   │   ├── map_10x10/
│   │   │   │   ├── map_aoba00/   # 青葉山（実地図）
│   │   │   │   ├── map_aoba01/
│   │   │   │   ├── map_kyodai/   # 京都大学
│   │   │   │   ├── map_osaka/    # 大阪
│   │   │   │   ├── map_paris/    # パリ
│   │   │   │   ├── map_shibuya/  # 渋谷
│   │   │   │   └── map_shijo/    # 四条
│   │   │   ├── state_repre/      # 状態表現
│   │   │   │   ├── coordinate.py # 座標ベース
│   │   │   │   ├── onehot.py     # one-hot エンコーディング
│   │   │   │   └── onehot_fov.py # one-hot + 視野フィルタ（部分観測）
│   │   │   └── wrapper/
│   │   │       ├── safe_marl.py  # SafeEnv：無効行動を防ぐラッパー
│   │   │       └── drp_task.py   # TaskEnv：タスク管理ラッパー
│   │   └── assets/markdown/      # 補足ドキュメント
│   │
│   └── epymarl/                  # MARL 学習フレームワーク（Extended PyMARL）
│       └── src/
│           ├── main.py           # 学習エントリポイント
│           └── config/algs/      # IQL, QMIX, MAPPO, etc. の設定ファイル
│
└── logs/                         # run.py の実験ログ出力先
```

---

## 4. インストール方法

### 前提条件

- Python 3.9
- Conda（推奨）または venv

### 手順

```bash
# 1. リポジトリをクローン
git clone https://github.com/kaji-ou/LDRP.git
cd LDRP

# 2. 仮想環境を作成（Python 3.9 を推奨）
conda create -n ldrp python=3.9
conda activate ldrp

# 3. Gym 環境をインストール
pip install -e ./src/main/LDRP

# 4. 依存ライブラリをインストール
pip install -r ./src/main/requirements.txt

# 5. MARL 学習が必要な場合は epymarl の依存もインストール
pip install -r ./src/epymarl/requirements.txt
```

### 主な依存ライブラリ

| ライブラリ | バージョン | 用途 |
|------------|------------|------|
| `gym` | 0.26.2 | 環境フレームワーク |
| `networkx` | 3.2.1 | グラフ操作（マップ管理） |
| `matplotlib` | 3.8.2 | 可視化 |
| `torch` | — | MARL モデルの推論・学習 |
| `numpy` | — | 数値計算 |
| `pyyaml` | — | 設定ファイルの読み込み |

---

## 5. 実行方法

### 5-1. 単一テスト実行（test.py）

```bash
# 引数なし（default.yaml の設定を使用）
python test.py

# 引数あり：map, エージェント数, 経路計画, タスク割り当て
python test.py <map_name> <agent_num> <path_planner> <task_assigner>

# 具体例
python test.py map_3x3 3 pbs tp       # PBS + Task Priority, 3エージェント
python test.py map_8x5 4 iql fifo     # IQL + FIFO, 4エージェント
python test.py map_aoba00 5 qmix tp   # QMIX + TP, 5エージェント（青葉山マップ）
```

#### 引数の選択肢

| 引数 | 選択肢 |
|------|--------|
| `map_name` | `map_3x3`, `map_5x4`, `map_8x5`, `map_10x6`, `map_10x8`, `map_10x10`, `map_aoba00`, `map_aoba01`, `map_kyodai`, `map_osaka`, `map_paris`, `map_shibuya`, `map_shijo` |
| `agent_num` | 1 〜 30 の整数 |
| `path_planner` | `pbs`, `iql`, `qmix` |
| `task_assigner` | `fifo`, `tp`, `ppo` |

### 5-2. まとめて実験（run.py）

複数のマップ・エージェント数・アルゴリズムの組み合わせを一括実行します（最大 5 並列プロセス）．

```bash
python run.py
```

`run.py` の先頭部分で実験条件を設定します：

```python
map_name = ["map_5x4", "map_8x5", "map_aoba00", "map_aoba01"]
agent_num = [3, 4, 5]
path_planner = ["iql", "qmix"]
task_assigner = ["fifo", "tp"]
```

結果は `logs/<map_name>/safe/` 以下にテキストファイルとして保存されます．

### 5-3. インタラクティブ GUI テスト（drpload_test.py）

PBS アルゴリズムを視覚的にデバッグ・確認したい場合：

```bash
python drpload_test.py
```

matplotlib ウィンドウが開き，ステップごとにエージェントの動きを確認できます．

### 5-4. MARL モデルの学習（train.py）

```bash
python train.py
```

内部では `epymarl` の `main.py` を呼び出します．`train.py` の `command` 変数で条件を指定します：

```python
# 例：IQL で map_aoba00 を 4 エージェントで学習（10 回並列実行）
command = f'python3 src/epymarl/src/main.py \
  --config=iql \
  --env-config=gymma \
  with env_args.time_limit=100 \
  env_args.key="drp_env:drp-4agent_map_aoba00-v2" \
  env_args.state_repre_flag="onehot_fov" \
  > train_results/{i} 2>&1'
```

サポートされているアルゴリズム：`iql`, `qmix`, `vdn`, `mappo`, `coma`, `ia2c`, `maddpg` 等

---

## 6. 設定ファイルの詳細

### src/config/default.yaml

```yaml
# Gym 環境名（SafeEnv を使う場合は drp_safe を指定）
env_name: "drp_env:drp_safe-4agent_map_8x5-v2"

# エージェント数（test.py の引数で上書き可能）
agent_num: 3

# 使用するマップ名
map_name: "map_3x3"

# タスク割り当てアルゴリズム
# fifo: 順番割り当て
# tp:   最短距離ベースの貪欲法
# ppo:  学習ベース（PPO）
task_assigner: "tp"

# 経路計画アルゴリズム
# pbs:  Priority-Based Search（記号的探索）
# iql:  学習済み IQL モデル
# qmix: 学習済み QMIX モデル
path_planner: "pbs"

# 1 エピソードの最大ステップ数
time_limit: 1000

# テスト実行回数
test_num: 1

# --- 以下は学習用パラメータ ---
n_envs: 8              # 並列環境数
running_steps: 20000000  # 総学習ステップ数
buffer_size: 1024
batch_size: 64
epochs: 4
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
entropy_coef: 0.0001
value_loss_coef: 0.5
```

### gym.make の引数（test.py 内）

```python
env = gym.make(
    env_name,
    state_repre_flag="onehot_fov",  # 状態表現（coordinate/onehot/onehot_fov）
    reward_list=reward_list,         # 報酬設定
    time_limit=config.time_limit,    # エピソード最大ステップ数
    task_flag=True,                  # タスクあり環境を使う場合 True
    task_list=None,                  # 固定タスクリストを使う場合はここに渡す
)
```

---

## 7. アルゴリズムの解説

### 7-1. 経路計画アルゴリズム

#### PBS (Priority-Based Search)

記号的な優先順位ベース探索アルゴリズムです．強化学習なしで動作します．

**動作の流れ：**
1. 各エージェントに優先順位を割り当て（近いエージェントほど低優先）
2. 優先順位の高いエージェントから順に BFS で経路を探索
3. 先行エージェントの過去位置・現在位置を障害物として扱い，衝突回避
4. 経路が見つからない場合は優先順位を変えて再探索

**注意点：**
- ノードが密集したマップや自由度が低い環境では，解が見つからないことがある
- 他エージェントが道を塞ぐと行き詰まる場合がある

#### MARL ポリシー（IQL / QMIX）

`src/all_policy/models/` に保存された学習済みモデルを使用します．
- **IQL**: 各エージェントが独立した Q ネットワークで行動選択
- **QMIX**: 個別の Q 値を混合関数で統合（協調的な行動が可能）

### 7-2. タスク割り当てアルゴリズム

#### FIFO（random.py）

タスクリストの順番通りに空きエージェントへ割り当てます．最もシンプルなベースライン．

#### TP (Task Priority)

貪欲法による割り当てです：
- 空きエージェントごとに，最もピックアップ場所に近いタスクを割り当てる
- NetworkX の最短経路計算を利用

#### PPO（学習ベース）

PPO アクタークリティック NN がタスク割り当てを学習します．
- 経験バッファに基づく学習
- エンドツーエンドでタスク割り当て戦略を最適化

---

## 8. 環境の仕様

### マップ

CSV ファイルで定義されたグラフ構造です．

```
map_NAME/
├── node.csv    # ノードID, x座標, y座標, z座標, station
└── edge.csv    # 始点ノード, 終点ノード（無向グラフ，重みは距離で計算）
```

### 状態表現（state_repre_flag）

| フラグ | 内容 |
|--------|------|
| `coordinate` | 各エージェントの (x, y) 座標 |
| `onehot` | ノード占有状況の one-hot ベクトル |
| `onehot_fov` | one-hot + 視野フィルタ（部分観測，推奨） |

### 行動空間

- 離散行動空間：各ステップで「次に向かうノード番号」を選択
- 無効行動（移動不可能なノードへの指示）は「その場に留まる」扱い

### 報酬設定

| イベント | デフォルト値 | 変数名 |
|----------|-------------|--------|
| ゴール到達 | +100 | `reward_list["goal"]` |
| 衝突 | −100 | `reward_list["collision"]` |
| 待機 | −10 | `reward_list["wait"]` |
| 移動 | −1 | `reward_list["move"]` |

### Gym 環境名のパターン

```
drp_env:drp-{N}agent_{map_name}-v2        # 通常モード
drp_env:drp_safe-{N}agent_{map_name}-v2   # Safe モード（無効行動防止）
```

例：`drp_env:drp_safe-4agent_map_8x5-v2`（4 エージェント，8×5 マップ，Safe モード）

### タスク関連の環境変数

| 変数 | 内容 | 例 |
|------|------|----|
| `env.current_tasklist` | 未実行タスクのリスト（[ピックアップ, 配達] のリスト） | `[[1,2],[5,3],[8,9]]` |
| `env.assigned_list` | 各タスクがどのエージェントに割当済みか（−1 = 未割当） | `[1, 0, -1]` |
| `env.assigned_tasks` | 各エージェントに割当済みのタスク | `[[1,2],[3,4],[]]` |

### エピソードの流れ

```
env.reset()
     ↓
ループ（時間切れまで）：
  1. path_planner.policy(obs, env)  → 各エージェントの次ノード
  2. task_manager.assign_task(env)  → 各エージェントへのタスク割り当て
  3. joint_action = {"pass": [...], "task": [...]}
  4. obs, rew, done, info = env.step(joint_action)
     ↓
  タスク完了数，衝突数，ステップ数を集計
```

---

## 9. 独自 Policy の実装方法

`src/policy.py` の `Policy` クラスを参考に実装します．

### joint_action の形式

```python
joint_action = {
    "pass": [node_0, node_1, node_2, ...],  # 長さ = agent_num, 各値 = 移動先ノード番号
    "task": [task_idx_or_-1, ...]           # 長さ = agent_num, -1 = タスク割り当てなし
}
```

### task_assign の例

```python
# current_tasklist = [[1,2], [5,3], [8,9]] の場合
# エージェント0 に タスク[5,3] (index 1) を割り当て
# エージェント1 に タスク[1,2] (index 0) を割り当て
# エージェント2 には割り当てなし
task_assign = [1, 0, -1]
```

**重要：** タスクを割り当てられるのは「未実行状態（タスクを持っていない）」のエージェントのみです．

### PolicyManager / TaskManager の利用

既存のアルゴリズムを使う場合：

```python
from src.all_policy.policy_manager import PolicyManager
from src.task_assign.task_manager import TaskManager

path_planner = PolicyManager(args)   # args.path_planner = "pbs" / "iql" / "qmix"
task_manager = TaskManager(args.task_assigner, args)  # "fifo" / "tp" / "ppo"

agents_action = path_planner.policy(obs, env)
task_assign = task_manager.assign_task(env)
joint_action = {"pass": agents_action, "task": task_assign}
```

---

## 10. 学習・モデル適用方法

### 学習済みモデルの適用

1. epymarl で学習したモデル（`.th` ファイル）を `src/all_policy/models/` に配置
2. `src/all_policy/policy.py` 内の `MARLPolicy` クラスでモデルのパスを指定
3. `default.yaml` の `path_planner` を `"iql"` または `"qmix"` に変更
4. `test.py` を実行

### epymarl での学習コマンド例

```bash
# IQL，4エージェント，map_8x5，onehot_fov 状態表現
python3 src/epymarl/src/main.py \
  --config=iql \
  --env-config=gymma \
  with env_args.time_limit=100 \
       env_args.key="drp_env:drp_safe-4agent_map_8x5-v2" \
       env_args.state_repre_flag="onehot_fov"

# QMIX，4エージェント，map_aoba00（実地図）
python3 src/epymarl/src/main.py \
  --config=qmix \
  --env-config=gymma \
  with env_args.time_limit=100 \
       env_args.key="drp_env:drp-4agent_map_aoba00-v2" \
       env_args.state_repre_flag="onehot_fov"
```

---

## 11. 注意事項・既知の問題

### PBS について

- 「他エージェントが道を封鎖する」「自由度が低い」などの理由で，衝突しない経路が見つからないことがある
- 解が見つからない場合，エージェントはその場で停止する

### drp_env.py の変更点（MARL4DRP からの変更）

- PBS のために `drp_env.py` の 200 行目付近を変更している → **強化学習に影響を与える可能性あり**
- 継続型問題に対応するため，エージェントがゴールに到達した状態でも `avail_actions` がゴールノードに固定されないよう変更

### Safe モードについて

`drp_safe-` 環境名を使うと，SafeEnv ラッパーが有効になります：
- 同一ノードへの複数エージェントの移動を防止
- 正面衝突（エージェントが互いの位置を入れ替える動き）を防止
- 無効行動は「その場に留まる」に変換

### タスクの発生タイミング

- タスクは各ステップで `env.current_tasklist` に追加される（動的生成）
- いつ・どんなタスクが追加されるかは **エピソード開始時に決定** される
- 固定タスクリストを使いたい場合は `gym.make(..., task_list=[...])` で渡す

---

## まとめ

| 目的 | 実行ファイル |
|------|-------------|
| 動作確認・デバッグ | `python test.py` |
| 複数条件の比較実験 | `python run.py` |
| PBS の GUI 確認 | `python drpload_test.py` |
| MARL モデルの学習 | `python train.py` |

```
LDRP = DRP（経路計画）+ タスク割り当て + 継続的なタスク発生
     = MARL4DRP の機能拡張版
```
