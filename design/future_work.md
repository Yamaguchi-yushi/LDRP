# 将来実装メモ (TODO 集約)

軽量な「将来やりたい」「未適用の修正」を集約するファイル。重い独立設計書 (例: [ldrp_extensions.md](ldrp_extensions.md)) はここから参照のみ。

各項目は **背景 / 現状 / 対策案 / 影響範囲** の節構成で書く。実装に着手したら本ファイルから対応セクションを削除し、必要に応じて [../MANUAL.md](../MANUAL.md) の更新履歴に記録する。

---

## 目次

1. [LaRe-Path 因子の正規化 (3 因子)](#1-lare-path-因子の正規化-3-因子)
2. [LaRe-Path 距離因子の残課題 (エッジ補間精度・タスク切替時の prog_goal)](#2-lare-path-距離因子の残課題-エッジ補間精度タスク切替時の-prog_goal)
3. [MAT-Dec 学習済みモデルの評価実行 (test.py) 対応](#3-mat-dec-学習済みモデルの評価実行-testpy-対応)
4. [タスク発生分布の可変化 (到着レート制御)](#4-タスク発生分布の可変化-到着レート制御)
5. [評価シナリオの準備 (固定タスク列 + シナリオ集合)](#5-評価シナリオの準備-固定タスク列--シナリオ集合)
6. [エージェント稼働状態の指標化 (空走率 / idle 率)](#6-エージェント稼働状態の指標化-空走率--idle-率)
7. [既知の問題インデックス (2026-08-12 棚卸し)](#7-既知の問題インデックス-2026-08-12-棚卸し)
8. [学習スイープと実験管理 GUI](#8-学習スイープと実験管理-gui)
9. [非アクティブ機の観測と衝突判定の不整合](#9-非アクティブ機の観測と衝突判定の不整合)
10. [検出済みの学習条件の食い違い](#10-検出済みの学習条件の食い違い)

### 重い設計書 (別ファイル)

- [ldrp_extensions.md](ldrp_extensions.md): LDRP 拡張 (高優先度: ピックアップ前タスク再配布 / 低優先度: 複数タスク保持 = VRP/TSP 系 tour 計画)
- [env_maturity.md](env_maturity.md): 環境ソフトウェアとしての成熟度ギャップ (CAMAR/RHCR/LoRR 比較。評価プロトコル・回帰テスト・throughput 指標・大規模輻輳耐性など)
- [dynamic_agent_count.md](dynamic_agent_count.md): タスク割当エージェントによる動的エージェント数制御 (Phase A: 実行時のみ増減 → Phase B: 増減も学習)

---

## 1. LaRe-Path 因子の正規化 (3 因子)

### 背景

[CLAUDE.md](../CLAUDE.md) の不変条件「LaRe-Path の 10 因子は概ね [0, 1] に正規化」に対し、3 因子が **未正規化のまま** デコーダ MLP に渡されている。スケール差が大きい因子だけで proxy 報酬が予測される縮退状態に陥り、他因子の情報が学習に乗らないリスクあり。

### 現状の値域 ([encoder.py](../src/lare/path/encoder.py) より)

| # | 因子 | 計算式 | 実際の値域 | 状態 |
|---|---|---|---|---|
| 1 | `prog_goal` | dist_prev - dist_curr | **[-D, +D]** (D=graph_diameter ≈ 200) | ❌ |
| 2 | `in_collision` | 0 or 1 | {0, 1} | ✓ |
| 3 | `others_in_collision` | 0 or 1 | {0, 1} | ✓ |
| 4 | `wait_norm` | wait_count (連続カウント、リセット済み) | **[0, time_limit]** 理論上限。実態は数十 step 程度に収束 | ❌ |
| 5 | `dist_goal_norm` | dist / D | [0, 1] | ✓ |
| 6 | `min_sep_norm` | min_sep / D | [0, 1] | ✓ |
| 7 | `avg_sep_norm` | avg_sep / D | [0, 1] | ✓ |
| 8 | `safety_margin` | min_sep / collision_dist, clip(0, 100) | **[0, 100]** | ❌ |
| 9 | `collision_risk` | 1 if min_sep < coll_dist*2 else 0 | {0, 1} | ✓ |
| 10 | `at_goal` | 1 if dist < eps else 0 | {0, 1} | ✓ |

### 対策案

#### prog_goal (距離変化)

進捗の **符号情報**は actor の学習に効くので、ただ clip するより向きを残したい。

| 案 | 式 | 範囲 | 評価 |
|---|---|---|---|
| A | `prog_goal / D` | [-1, +1] | 情報損失なし。他因子 [0,1] と符号スケールが微妙 |
| B | `(prog_goal / D + 1) / 2` | [0, 1] | 0.5 を「中立」とするシフト。他因子と完全に揃う |
| C | `clip(prog_goal / D, 0, 1)` | [0, 1] | 「進んだ量だけ評価」。MARL4DRP がこれなら踏襲 |

→ **MARL4DRP の参照実装 (`marl4drp-lookup` subagent) で揃え先を確認するのが安全**。

#### wait_norm (連続 wait 回数。リセット動作は既に [drp_env.py:759](../src/main/drp_env/drp_env.py#L759) で適用済み、残るは正規化のみ)

| 案 | 分母 | 範囲 | 評価 |
|---|---|---|---|
| A | `time_limit` (500) | [0, 1] | エピソード長依存 |
| B | 固定定数 (例: 20〜50) | [0, 1] clip | 「N step 以上待っている = 完全に詰まっている」のセマンティクス |
| C | `1 - exp(-w / τ)` (τ=5〜10) | [0, 1) | 少数回 wait に強く反応、長期 wait は飽和 |

#### safety_margin

`collision_risk` (#9) が既に「margin < 2 でアラート」を 0/1 で出しているので、本因子の本質は「衝突距離の何倍離れているか」の連続値情報。margin > 1 (衝突距離より遠い) は全部 1 扱いでも情報損失少ない可能性。

| 案 | 式 | 範囲 | 評価 |
|---|---|---|---|
| A | `clip(margin, 0, 1)` | [0, 1] | margin > 1 は全部 1。情報潰れる |
| B | `min(margin / K, 1)` (K=5〜10) | [0, 1] | 中間域の情報を保つ |
| C | `1 - exp(-margin / τ)` | [0, 1) | 近距離で 0、遠距離で漸近的に 1 |

### 影響範囲

- [src/lare/path/encoder.py](../src/lare/path/encoder.py): 3 因子の正規化式変更
- 既存学習済みモデル (`.pth`) は **デコーダ入力スケールが変わるので再学習が必要**

---

## 2. LaRe-Path 距離因子の残課題 (エッジ補間精度・タスク切替時の prog_goal)

> **前提 (実装済み)**: エッジ上の位置を partial onehot で `obs_onehot` に温存する仕様を採用済み ([drp_env.py:999-1000](../src/main/drp_env/drp_env.py#L999-L1000))。これにより `estimate_partial_distance` の 2 要素分岐 ([encoder.py:96-113](../src/lare/path/encoder.py#L96-L113)) が機能し、エッジ上の移動が `prog_goal` / `dist_goal_norm` に反映されるようになった。以下はその上に残る精度・定義上の課題。

### 残課題 A: `estimate_partial_distance` がエッジ長 L を無視

2 要素分岐 ([encoder.py:100-109](../src/lare/path/encoder.py#L100-L109)) の距離は `(1-α)·Di + α·Dj` の **線形補間で、エッジ長 L を距離に加算していない**。エッジ中央で最大 `(L²−(Di−Dj)²)/(2L)` (= 等距離端点で 0.5L) 過小評価する。`|Di−Dj|≈L` (ゴール方向に直進) のときは誤差ゼロなので、**長い横向き (迂回) エッジが多いマップでのみ実害**が出る。優先度低 (現状の量子化ノイズに埋もれるレベル)。

**対策案**: 端点経由の min ルーティングに変更 (`graph` の隣接重み = エッジ長 L を利用):

```python
# 正規化重み wi_n, wj_n (= 各ノードへの近さ), L = edge_length(i, j)
#   点→端点の距離はエッジ長 L に比例: d_to_i = wj_n*L, d_to_j = wi_n*L
return min(wj_n * L + Di, wi_n * L + Dj)
```

L が取れない (隣接でない) 異常ケースは従来の線形補間にフォールバック。partial 仕様とは独立な精度改善。

### 残課題 B: タスク切替 step の prog_goal リセット

**背景**: `prog_goal = dist_goal_prev - dist_goal` は両項とも **現在のゴール基準** で計算される ([encoder.py:129-139](../src/lare/path/encoder.py#L129-L139))。ゴールが切り替わった step では `dist_goal_prev` が「**前 step の位置から“新しい”ゴールまでの距離**」になり、前 step に存在しなかった目標に対する差分 = 意味のないクロス目標値になる。

特に害が大きいのが **ピックアップ到達 step**。ドロップ D が来た道方向 (戻る側) にあると:

- 前 step: X→P へ前進 (当時のゴール P へは正しく進捗)
- この step: ゴールが D に切替、`dist_goal_prev = dist(X, D)` 小、`dist_goal = dist(P, D)` 大
- → `prog_goal = 小 − 大 = 大きな負`

ピックアップ成功という good event の瞬間に大きな負の進捗が出て、デコーダに誤信号を与える。新規割当 (idle→pickup) では agent が動いていないので `prog_goal ≈ 0` になりやすく、害は主に「移動しつつゴールが変わる」遷移 (= ピックアップ到達) で出る。

**対策案**: ゴールをまたいだ進捗は **定義不能** なので、その step は中立値に上書きする。

- **`prog_goal` だけ** を中立値にする。`dist_goal_norm` / `at_goal` は「現在状態の量」で新ゴール基準でも正しいのでそのまま残す。
- **中立値は項目 1 の正規化選択に合わせる**: raw / 案A (`prog/D`) / 案C (`clip(prog/D,0,1)`) → **0**。案B (`(prog/D+1)/2`) → **0.5**。
- **検出は env 側が確実**: タスクブロックで `goal_array[i]` が変わるのを env は知っている。`compute_factors` に `goal_changed` マスクを渡し、encoder 側で該当 agent の `prog_goal` を中立値に潰すのがクリーン。
- **実装**: prev onehot を退避する `_lare_capture_prev_onehot_pos` ([drp_env.py:603](../src/main/drp_env/drp_env.py#L603)) で前 step のゴール (`goal_array` のスナップショット) も 1 本並行保持し、`prev_goal[i] != curr_goal[i]` を判定するだけ。

**影響範囲**:

- `_lare_capture_prev_onehot_pos` にゴールスナップショットを追加 + `compute_factors`/`evaluation_func` に `goal_changed` マスク経路を 1 本追加
- `prog_goal` の分布が変わるため **再学習推奨** (項目 1 と同時適用が望ましい)
- `use_lare_path=False` の挙動は不変

---

## 3. MAT-Dec 学習済みモデルの評価実行 (test.py) 対応

### 背景

MAT-Dec ([src/epymarl/src/config/algs/mat_dec.yaml](../src/epymarl/src/config/algs/mat_dec.yaml)) で学習した方策を `test.py` で評価したい。actor の実体 `Decoder` ([mlp_mat_agent.py:100-122](../src/epymarl/src/modules/agents/mlp_mat_agent.py#L100-L122)) は全エージェント重み共有の MLP で `n_agents` に依存する重みを持たないため、**学習時と異なるエージェント数への汎化評価**にも使える (この汎化検証が主目的)。

### 現状

- 推論パイプライン ([src/all_policy/](../src/all_policy/)) は RNNAgent 専用:
  - `PolicyRunner.__init__` ([policy_runner.py:23-26](../src/all_policy/policy_runner.py#L23-L26)) が state_dict に `"fc1.weight"` キーを要求し、MAT-Dec の checkpoint (`decoder.mlp.0.weight` 等) は即 ValueError で落ちる
  - アーキテクチャ自動検出 (`use_rnn` 判定・input_shape 読取) も RNNAgent のキー名前提
- 保存側は QMIX/IQL 系と同じ `basic_controller.save_models()` → `agent.th` (state_dict) なのでファイル形式自体は流用可能
- **`agent.th` には critic (Encoder) の重みも同居する**: `mat_learner.py` の `self.mac.agent.critic = self.critic` ([mat_learner.py:25](../src/epymarl/src/learners/mat_learner.py#L25)) で critic が agent の属性としてアタッチされ、`nn.Module` の属性代入で自動サブモジュール登録されるため。推論時は `decoder.*` だけ使い `critic.*` は無視してよい

### 対策案 (設計確定・実装は未着手)

1. **`src/all_policy/mat_policy_runner.py` を新設**。`Decoder` 相当のクラス (LayerNorm → Linear → GELU ×2 → Linear, `mlp_mat_agent.Decoder` と同一の層構成) をローカル再実装し、`agent.th` の `decoder.` プレフィックス付きキーだけを抜き出して `load_state_dict`。`critic.*` キーは無視。RNNAgent と違い hidden state 管理は不要 (`use_rnn=False` でステートレス)
2. **次元の自動検出**: `decoder.mlp.1.weight` (最初の Linear, shape=`(n_embd, obs_dim)`) と `decoder.mlp.7.weight` (最後の Linear, shape=`(n_actions, n_embd)`) の shape から `obs_dim`/`n_embd`/`n_actions` を復元し、env 側の `input_shape`/`n_actions` と食い違えば ValueError (既存 `PolicyRunner` の自動検出と同じ思想)
3. **クラスの出し分けは `path_planner` の明示分岐**: state_dict のキー自動判別ではなく、`MARLPolicy.policy()` で `self.path_planner in {"mat_dec"}` のときだけ `MatPolicyRunner` を使う (それ以外は既存 `PolicyRunner`)。出力の意味 (Q値 vs 方策logits) がアルゴリズムごとに違うため、キー名だけで自動判別するより明示的な方が安全
4. **行動選択は決定的 argmax**: 出力 (方策logits) を `avail_actions` (既存の index リスト形式のまま、`policy.py` 側の呼び出しインタフェースは変更不要) でマスクし `-1e10` → argmax。既存 `PolicyRunner` の masked-Q argmax と全く同じ書き方に揃える (`logit[a] if a in avail_actions else -1e10` の形)。確率的 sample にはしない (QMIX系評価と挙動を揃え、エピソード間の再現性を優先)
5. **`MARLPolicy.get_model_path`** ([policy.py:42-48](../src/all_policy/policy.py#L42-L48)) の命名規則 `{map}_{N}_{algo}.th` はそのまま `mat_dec` を algo 名として使える (変更不要)
6. **自己回帰 decode は評価時不要**: `discrete_autoregreesive_act` ([mlp_mat_agent.py:61-84](../src/epymarl/src/modules/agents/mlp_mat_agent.py#L61-L84)) は critic (`v_loc`) と絡むが、行動選択自体は per-agent の `decoder(obs)` → argmax で完結する。critic はロード不要

参考: epymarl 自身の eval (train.py 実行中の sacred テストエピソード) は `SoftPoliciesSelector` ([action_selectors.py:67-75](../src/epymarl/src/components/action_selectors.py#L67-L75)) が `test_mode` を見ずに常時サンプリングするため、上記の決定的 argmax とは一致しない。test.py 側の評価はあくまで独自の決定的ポリシーとして扱う。

### 汎化評価時の注意

- `n_embd` 等のハイパーパラメータは checkpoint の shape から復元可能だが、`state_repre_flag='onehot_fov'` の obs 次元はマップサイズ・FOV に依存するため、**汎化はエージェント数方向のみ** (マップをまたぐ汎化は obs 次元が変わり不可)
- `obs_agent_id=True` で学習した checkpoint は入力にエージェント数分の onehot が付くため汎化不可。mat_dec.yaml のデフォルトは `obs_agent_id: False`

### 影響範囲

- `src/all_policy/` に新規ファイル追加 + `policy.py` or `policy_manager.py` に分岐数行
- 既存の RNN 系評価パスは不変

---

## 4. タスク発生分布の可変化 (到着レート制御)

### 背景

現状のタスク生成 [create_tasklist](../src/main/drp_env/EE_map.py#L295) は **毎ステップ必ず 1 個**をハードコードで発生させる (`random_num = 1`)。引数 `task_density` は渡されるが未使用。time_limit=1000 なら 1000 個の到着スケジュールになり、**到着レートが高すぎて** `current_tasklist` が常時 `task_num`(=agent_num×2) 満杯 → 超過タスクは [drp_env.py:905](../src/main/drp_env/drp_env.py#L905) の `if len(self.current_tasklist) < self.task_num` で**捨てられる**。これは (a) 割当方策の学習信号を汚し、(b) 疎/密の状況差を作れない。

さらに [aamas_submission.md §5.1](aamas_submission.md) の**疎/密/バースト到着シナリオ**は、この生成レート制御が実装前提。→ 論文実験のインフラでもある。

### 現状

- `create_tasklist(timelimit, agent_num, task_density)`: `task_density` 未使用、毎ステップ 1 個固定。
- 呼び出し元 [drp_env.py:670](../src/main/drp_env/drp_env.py#L670): `create_tasklist(self.time_limit, self.agent_num, 1)` (density=1 をハードコード)。

### 対策案 (設計確定・実装は未着手)

`task_density` を実際に使い、確率的到着を導入する。**mode パラメータで分布を切替**、デフォルトは従来動作 (`fixed`) で **baseline 不変条件を維持**する。

各ステップ `t` に発生するタスク数を `N_t`（確率変数）とする。`create_tasklist` は `t=0..T-1` について `N_t` 個のタスクを生成する。

| mode | 各 step の発生数 `N_t` | 用途 |
|---|---|---|
| `fixed` (デフォルト) | `N_t = 1` (定数) | 従来動作・baseline 互換 |
| `bernoulli` | `N_t ∈ {0,1}`, `P(N_t=1)=p` | 高々 1/step の疎化。間隔が確率的 |
| `poisson` | `N_t ~ Poisson(λ)` | 0 や複数も許す平均レート制御 |
| `mmpp` | 状態依存レート `p(z_t)` の Bernoulli | **密/疎の時期が確率的に交替 (時変)** |
| `scheduled` | フェーズ依存レート `p(⌊t/L⌋)` | 密/疎を決め打ち区間で交替 (再現性◎) |

#### (1) 定常レート: Bernoulli / Poisson

**Bernoulli(p)** — 各 step 独立に確率 `p` で 1 個。

```
N_t ~ Bernoulli(p),   E[N_t] = p,   タスク間隔 ~ 幾何分布(平均 1/p)
```

明示的な「遅延」ロジックを書かなくても、**間隔が幾何分布になるので「確率的な遅延をおいて発生」する挙動**になる。

```
p = 0.3 のイメージ (● = タスク発生)
step:  1  2  3  4  5  6  7  8  9 10 11 12
       -  ●  -  -  ●  -  -  -  ●  -  ●  -
          └─3─┘   └───4───┘   └2┘        ← 間隔が確率的に変動
```

**Poisson(λ)** — 1 step に複数発生も許す (`λ` = 平均到着数/step)。`λ<1` で疎、`λ>1` で密。

```
N_t ~ Poisson(λ),   P(N_t=k) = e^{-λ} λ^k / k!,   E[N_t] = λ
```

疎/密の目安 (10 台 aoba00): 疎 `p,λ ≈ 0.1–0.2` / 中 `≈ 0.3–0.5` / 密 `≈ 0.7–1.0`。上限は `task_num=agent_num×2` の破棄が慢性化しない範囲で調整。

#### (2) 時変レート: 密な時期 / 疎な時期 (MMPP / スケジュール)

「時間帯で密/疎が変わる」= 非定常到着。標準的には **非斉次ポアソン過程 (NHPP, レート `λ(t)`)** や **マルコフ変調ポアソン過程 (MMPP, 状態でレート切替)**。LDRP では Bernoulli レート `p` を時変にするだけで実現する。

**mmpp (確率的な密/疎の交替)** — 隠れ状態 `z_t ∈ {dense, sparse}` をマルコフ連鎖で遷移させ、状態ごとにレートを変える。

```
状態遷移: 各 step 確率 q で状態を反転      z_{t+1} = flip(z_t) w.p. q,  else z_t
レート  : p(z_t) = p_high  (z_t = dense)
                  p_low   (z_t = sparse)
発生    : N_t ~ Bernoulli( p(z_t) )

平均滞在時間: dense/sparse とも 1/q ステップ (幾何分布)
時間平均レート: (p_high + p_low) / 2   (対称遷移のとき)
```

```
p_high=0.8, p_low=0.1, q=1/150 のイメージ
        ┌── dense ──┐        ┌──── dense ────┐
z_t : ..dense dense..sparse sparse sparse..dense dense..
N_t : ●●●● ●● ● ●   ·    ·    ·   ·     ●● ●●● ● ●●
      ↑密集(高レート)      ↑まばら(低レート)      ↑また密集
```

**scheduled (決め打ち区間で交替)** — 区間長 `L` ごとに密/疎を交替 (再現性・可視化しやすい)。

```
phase = ⌊t / L⌋ mod 2
p(t)  = p_high (phase=0) / p_low (phase=1)

  |<-- L -->|<-- L -->|<-- L -->|
  |  dense  |  sparse |  dense  |   ...
   ●●●●●●●●   ·  ·  ·   ●●●●●●●●
```

#### パラメータと実装ノート

- 追加 env パラメータ: `task_arrival` (str, デフォルト `"fixed"`), `task_density` (float = `p`/`λ`)。時変系は `p_high, p_low, switch_prob q`(mmpp) / `phase_len L`(scheduled) を追加。
- signature デフォルトで従来一致 (= [CLAUDE.md](../CLAUDE.md) の baseline 不変条件)。
- 再現性: `create_tasklist` に `rng` を渡し、seed 管理下で分布を振る。**scheduled は決め打ちなので密/疎区間を実験図に明示できる**。
- **バースト**は mmpp の特別形 (`p_high≈1, p_low≈0, q` 小) or Poisson の `λ` を一時的に跳ね上げる形で表現可。

### 影響範囲

- [EE_map.py](../src/main/drp_env/EE_map.py) の `create_tasklist` に分岐追加。
- [drp_env.py](../src/main/drp_env/drp_env.py) の `__init__` に 2 パラメータ + 呼び出し 1 箇所。
- `task_arrival="fixed"` の限り既存挙動・既存学習済みモデルとの互換は不変。
- 疎化すると超過破棄が減り、割当方策 (PPO / LaRe-Task) の学習信号がクリーンになる副次効果。
- **AAMAS 実験との関係**: 疎/密/バーストの 3 シナリオがこの 1 パラメータ群で構成できる ([aamas_submission.md §5.1](aamas_submission.md))。

---

## 5. 評価シナリオの準備 (固定タスク列 + シナリオ集合)

### 背景

公平な比較 (再割当 OFF/ON, 固定 N vs 動的, アルゴリズム間) には **全条件が同一のタスク列**を見る必要がある。seed 固定でも、条件間で乱数消費がズレる (エージェント数が違う / 方策が確率的) とタスク列が変わり不公平になる ([§4](#4-タスク発生分布の可変化-到着レート制御) 参照)。→ **タスク列を事前生成して固定シナリオ化**するのが最も確実 (LoRR / RHCR の task-file 方式)。[aamas_submission.md §5.4](aamas_submission.md) の統計・再現性の土台。

### 現状 (既に半分できている)

- タスク列は `self.alltasks` として保持され、**`task_list` コンストラクタ引数で外から渡せる** ([drp_env.py:673](../src/main/drp_env/drp_env.py#L673) `if self.alltasks is None:` → 渡されていれば生成をスキップ)。
- **形式**: `alltasks[t]` = ステップ `t` に発生するタスクのリスト。各タスク = `[start_node, goal_node, deadline]` (deadline は現状未実装で `time_limit+1`)。長さ = `time_limit`。
- **制約**: 現状 `alltasks` は1回セットしたら**全エピソード使い回す** → 「同一シナリオの反復」になり、決定的方策では全エピソード同結果 (分散ゼロ)。統計を取るには複数シナリオ対応が要る (下記)。

### 対策案 (設計確定・実装は未着手)

#### 5.1 シナリオ形式

```
alltasks: 長さ time_limit のリスト
alltasks[t] = [[s, g, deadline], ...]   # step t に発生するタスク群 (0 個も可)
例 (time_limit=6):
[ [], [[4,2,7]], [], [[7,0,7],[3,9,7]], [], [[1,5,7]] ]
```

JSON で保存し (`scenarios/<name>.json`)、評価時に読み込んで `task_list` に渡す。

#### 5.2 生成 (タスク専用 RNG + 到着分布 + 同ノード間隔)

- **タスク専用 RNG** (`np.random.RandomState(task_seed)`) で生成し、エージェント位置・方策の乱数と**分離** → N やアルゴを変えても**同一シナリオ**になる ([§4](#4-タスク発生分布の可変化-到着レート制御) の `rng` 分離)。
- 到着分布は [§4](#4-タスク発生分布の可変化-到着レート制御) の mode (bernoulli / poisson / mmpp / scheduled) を流用。
- **同じノードへの二重発生回避 (任意)**: 「前タスクが運ばれてから次」は実行時 (方策依存) の事実なので固定シナリオには厳密には入らない。**生成時に想定ピック時間 `min_gap` ステップ以内に使った pickup ノードを避ける**近似で、再現性を保ったまま二重スタックを実質回避する。

```python
def make_scenario(time_limit, n_nodes, mode="bernoulli", p=0.3, seed=0, min_gap=30):
    rng = np.random.RandomState(seed)       # タスク専用 RNG (他の乱数と分離)
    last_used = {}                          # pickup node -> 最後に発生させた step
    alltasks = []
    for t in range(time_limit):
        tasks = []
        n = 1 if (mode == "bernoulli" and rng.random() < p) else 0
        for _ in range(n):
            free = [x for x in range(n_nodes)
                    if t - last_used.get(x, -10**9) >= min_gap]   # 二重発生回避
            if not free:
                continue
            s = int(rng.choice(free))
            g = int(rng.choice([x for x in range(n_nodes) if x != s]))
            tasks.append([s, g, time_limit + 1])
            last_used[s] = t
        alltasks.append(tasks)
    return alltasks
```

#### 5.3 シナリオ集合 (多様性 × ペア設計)

- **集合** = {疎, 中, 密, バースト} × {seed 0..K}（例 4 レジーム × 10 seed = 40 本）を `scenarios/` に保存。
- **全条件で同じ集合を評価 (ペア)**: 固定 N 各値 / ルール増減 / 学習増減 を**同一シナリオ列**で回す → タスク運の分散が相殺され、少ない本数で差を検出。
- **報告**: レジームごとに mean ± 95%CI、パレート図に載せる。各レジーム 10〜30 本で CI は十分絞れる (`√30` で SE ≈ σ の 18%)。「同一シナリオを膨大に反復」より「多様シナリオをペアで」の方が査読で強い。

#### 5.4 env のシナリオ集合対応 (小改修)

現状の「`alltasks` を1回だけセット」を、**エピソードごとにシナリオを切替**られるようにする:

- `task_list` に**シナリオのリスト** (or ディレクトリ) を許容し、`reset()` で `episode_idx % len(scenarios)` 番を `self.alltasks` にセット。
- 1 シナリオだけ渡した場合は従来どおり (全エピソード同一)。

### 影響範囲

- 新規: シナリオ生成ツール (`make_scenario` + 一括生成スクリプト) と `scenarios/` ディレクトリ。
- [drp_env.py](../src/main/drp_env/drp_env.py) `reset()`: 複数シナリオを episode index で切替 (数行)。単一 / None 時は従来動作維持 = **baseline 不変**。
- 評価側 (test.py / runner.py): シナリオ集合の読み込みとエピソード割当。
- **公平性**: 全条件が同一シナリオ集合 → seed 消費のズレに依らず完全再現。フリートサイジング主実験 ([dynamic_agent_count.md §6](dynamic_agent_count.md)) の前提インフラ。
- 関連: [§4 タスク発生分布](#4-タスク発生分布の可変化-到着レート制御)（到着プロセス本体）, [aamas_submission.md §5.1/§5.4](aamas_submission.md)（疎/密/バースト実験・統計）。

---

## 6. エージェント稼働状態の指標化 (空走率 / idle 率)

### 背景

AAMAS の主張では **「割当の質」と「台数の適切さ」を別々の指標で示す**必要がある ([aamas_submission.md §5.1](aamas_submission.md))。教授ミーティング (2026-08-03) でも「完了タスク数だけだと台数が多いほど得になる」という指摘があり、稼働状態の内訳が必要になった。

役割分担は以下で確定:

| 役割 | 指標 | 何を示すか |
|---|---|---|
| **割当品質** | 1 タスクあたり空走ステップ (主) / 空走率 (補助) | 近いエージェントを選べているか |
| **台数の適切さ** | idle 率 (= 1 − 稼働率) | 遊んでいる機体がどれだけあるか = 削減余地 |

**idle 率は主張の軸そのものではなく「なぜ台数を減らせたか」の説明変数**として使う。単独では「全部 off にすれば 0」でゲーム可能なので、必ずサービス水準 (完了数 / 待ち時間) と対にする。パレート図の 2 軸は `n_active_mean` × 完了数のまま。

同様に空走率も比率なので分母 (積載中時間) の影響を受ける。経路方策が下手だと積載中も伸びて**空走率が見かけ上下がる**ため、**絶対量である「1 タスクあたり空走ステップ」を主指標**にする。

### 現状

`drp_env.py` に以下は**実装済み** (2026-08-03):

- `self.active_agent_steps` — 毎ステップ稼働台数を累積 (`active` 未実装の間は `agent_num` と一致)
- `info["n_active_mean"]` — 平均稼働台数
- `info["task_completion_per_agent"]` — 稼働台数で正規化した完了数

**未実装**なのは稼働状態の内訳 (idle / 空車回送 / 積載中) と、それを使う派生指標。

なお `info` dict は [drp_env.py](../src/main/drp_env/drp_env.py) の `step()` 前半 (collision 判定の直前) で構築される一方、`self.task_completion += 1` はその後ろにあるため、**`info["task_completion"]` は「このステップの完了を含まない」1 ステップ遅れの値**になっている。全条件に等しく効くので比較自体は成立するが、`task_completion` で割る派生指標を増やすとずれが伝播する。

### 対策案 (設計確定・実装は未着手)

#### 6.1 稼働状態の 3 分割

`assigned_tasks[i]` と `goal_array[i]` の関係で判別できる。割当時に `goal_array[i] = assigned_tasks[i][0]` (ピック地点) が入り、ピック完了時に `assigned_tasks[i][1]` (ドロップ地点) へ切り替わることを利用する。

| 状態 | 判定 | 意味 |
|---|---|---|
| **idle** | `assigned_tasks[i] == []` | タスク無し |
| **空車回送 (deadhead)** | タスク有り かつ `goal_array[i] == assigned_tasks[i][0]` | ピック地点へ向かう = **割当が生んだ無駄** |
| **積載中 (loaded)** | タスク有り かつ `goal_array[i] == assigned_tasks[i][1]` | 運搬中 = **不可避な仕事** |

`busy = deadhead + loaded`。カウントは `active_agent_steps` と同じ位置 (`step_account += 1` の直後) で行い、両カウンタの増加回数を必ず一致させる。

#### 6.2 追加する info キー

| キー | 定義 | 用途 |
|---|---|---|
| `deadhead_steps_per_task` | 空車回送 agent-step / 完了数 | **割当品質の主指標** (低いほど良い) |
| `deadhead_ratio` | 空車回送 agent-step / busy agent-step | 補助 (分母の影響を受ける点に注意) |
| `busy_ratio` | busy agent-step / active agent-step | idle 率 = 1 − これ |
| `agent_steps_per_task` | busy agent-step / 完了数 | 1 タスクの総占有コスト |

数値キーなので [episode_runner.py](../src/epymarl/src/runners/episode_runner.py) が自動収集し、**epymarl 無改修で TensorBoard に `test_*_mean` として出る**。

#### 6.3 info の 1 ステップ遅れ修正

`return` 直前で `task_completion` 系のキーを再代入し、そのステップ内で発生した完了を反映させる。

#### 6.4 評価側の出力

[runner.py](../runner.py) の集計ブロックに idle 率 / 空走率 / 1 タスクあたり占有ステップを追加し、`test.py` の標準出力に出す。

### 影響範囲

- [drp_env.py](../src/main/drp_env/drp_env.py): `reset()` にカウンタ 2 個、`step()` に 1 行、稼働状態カウント用メソッド 1 個、`info` に 4 キー。**報酬・遷移・行動は変わらないので baseline 不変**。
- [runner.py](../runner.py): 集計ブロックに数行。
- 学習側は無改修 (info の数値キーが自動でログされる)。
- **Phase A との関係**: カウントを `getattr(self, "active", [True] * agent_num)` 経由にしておけば、[dynamic_agent_count.md §3](dynamic_agent_count.md) の `active` フラグ導入時に**書き換えなしで正しい値へ切り替わる**。
- **先行して測る価値**: idle 率は `active` 未実装でも固定 N で測れる。**N を振って idle 率が単調に上がることを示せれば、台数制御を導入する動機が実測で裏付く** ([aamas_submission.md §5.0](aamas_submission.md) のゲート判定にも直結)。
- 未確認: ピック地点とドロップ地点が同一ノードのタスクが生成されうる場合、空車回送と積載中の判別がつかない。値が不自然なら要確認。

---

## 7. 既知の問題インデックス (2026-08-12 棚卸し)

**位置づけ**: 2026-08-12 時点で判明している未解決の問題を**取りこぼさないための索引**。
数が多く一度に潰せないため、まず「何が残っているか」だけを 1 箇所に固定する。

**運用**: 詳細な差し替えコードを持つものは [next_actions.md](next_actions.md) 側にあるので、ここでは
**1 行の要約 + 参照先**に留める (両方に書くと必ず片方が古くなる)。参照先が無いものはここが唯一の記録なので、
着手するときに next_actions.md へ移してから作業する。解決したら本セクションから該当行を削除する。

### 7.1 実装バグ (優先度高)

| # | 問題 | 影響 | 参照 |
|---|---|---|---|
| A-1 | **`Buffer` が単一 env 前提**。`n_envs` を受け取りながら使っておらず、`compute_returns` が 1 本のリストを位置照合している | 同時学習 (32 並列) で**誤った return のまま静かに学習が進む**。例外にならないので気付けない | [next_actions.md §12-4](next_actions.md) に解決コードあり (実機検証済み) |
| A-2 | **`update()` のマスク不整合**。収集時はマスク後の分布から `log_prob` を取るのに、更新時は生 logits で `Categorical` を作っている | 別分布どうしで重要度比を計算しており **approx KL が意味を成さない**。学習は進むので気付けない | [next_actions.md §4-1](next_actions.md)。`add_actions` 呼び出し側に `mask` を渡す変更もセットで必要 |
| A-3 | **`test.py` の乱数制御が不完全**。`torch.manual_seed` が無く、`np.random.seed(0)` が固定値 | (a) PPO の `Categorical.sample()` は torch の乱数なので **seed を指定しても学習が再現しない**。(b) np が常に 0 なので **PPO 学習の replicate が全て同じタスク列**を見る (探索だけが違う = 部分的にしか独立でない) | [next_actions.md §3-4](next_actions.md) |
| ~~A-4~~ | ~~`train.py` が seed を渡していない~~ | **問題ではないと判断 (2026-08-13)**。sacred が run ごとに別 seed を自動生成し、値は run ディレクトリ名 (`qmix_seed236379847_...`) に残るため、独立な replicate と再現性の両方が既に成立している | — (対応不要) |
| A-5 | **RNN hidden state が評価時にリセットされない** | 数値影響は現状 MAPPO の 4 モデルのみ (QMIX/QPLEX は `use_rnn=False` で無害) だが、`use_rnn=True` にした瞬間に静かに壊れる | [next_actions.md §5](next_actions.md) |
| A-6 | **`PolicyRunner` の `hidden_dim` が 64 ハードコード** ([policy_runner.py:43](../src/all_policy/policy_runner.py#L43)) | 下表のとおり `hidden_dim=128` で学習したモデルが**読めない** | **本節が唯一の記録** (§5 完了に伴い next_actions から移設)。修正は 1 行 |
| A-7 | **`_accumulate_utilization()` が `is_tasklist` でガードされていない** ([drp_env.py:949](../src/main/drp_env/drp_env.py#L949)) | `task_flag=False` の env で `assigned_tasks` が `[]` のままなので `IndexError` で停止。**PBS が実行不能** (内部で 1 agent env を `task_flag=False` で作るため)。他の 3 箇所は `is_tasklist` でガード済みで、ここだけ漏れている | **本節が唯一の記録** (2026-08-25 発見)。修正は 3 行 (下記) |

**A-6 の詳細** (2026-08-14 に next_actions §5 から移設):

| モデル | hidden_dim | 読込 |
|---|---|---|
| `map_8x5_3/4/5_qmix_base.th` (3 本) | 128 | ❌ |
| `map_8x5_4_qmix_ours_base.th` | 64 | ✅ |
| `map_aoba00_*` (全部) | 64 | ✅ |

`use_rnn` は既にチェックポイントから自動判別している (`rnn.weight_ih` の有無) ので、同じ要領で **`state_dict["fc1.weight"].shape[0]` から `hidden_dim` を取れる**。`DummyArgs(hidden_dim=64, ...)` の固定値を置き換えるだけの 1 行修正。

**A-7 の詳細** (2026-08-25 に PBS 実行で再現):

```text
File "src/all_policy/pbs.py", line 151, in culc_actions
    obs, reward, done, info = self.env.step([action])
File "src/main/drp_env/drp_env.py", line 952, in _accumulate_utilization
    task = self.assigned_tasks[i]
IndexError: list index out of range
```

PBS は経路計画のシミュレーション用に **内部で 1 エージェントの env を持つ** ([pbs.py:26-32](../src/all_policy/pbs.py#L26-L32))。
これは `task_flag=False` で作られるため、`assigned_tasks` は [166 行](../src/main/drp_env/drp_env.py#L166) の `[]` のまま
(per-agent リストへの展開は `reset()` の `if self.is_tasklist:` ブロック内 = [787 行](../src/main/drp_env/drp_env.py#L787) だけ)。
稼働率指標 (§6) を追加したときのガード入れ忘れで、`assigned_tasks` を触る他の箇所
([1002](../src/main/drp_env/drp_env.py#L1002) / [922](../src/main/drp_env/drp_env.py#L922) / [1164](../src/main/drp_env/drp_env.py#L1164)) は
すべて `is_tasklist` でガードされている。

**修正**: `_accumulate_utilization()` の `active = getattr(...)` の直前に 3 行。

```python
		# task_flag=False の env (PBS が内部で使う 1 agent env 等) では
		# assigned_tasks が [] のままなので集計自体を行わない
		if not self.is_tasklist:
			return
		active = getattr(self, "active", [True] * self.agent_num)
```

**確認済み**: この修正を当てると `map_8x5` / 3 agent / PBS / TP が 2 エピソード完走し、`[RESULT]` 行も正しく出る
(複数 seed 評価の配線も PBS で機能する)。なお PBS は探索アルゴリズムで seed の概念が無いため、
`model_seed` を振っても結果は同一。std を取る対象ではなく**決定的なベースライン**として扱う。

### 7.2 未着手の実装 (next_actions.md に計画あり)

| # | 項目 | 状態 | 参照 |
|---|---|---|---|
| B-1 | MAT (完全版) の実装 | コード検証済み (学習発火 + N=3→5/8 ゼロショット) だが**ファイル未作成**。§9・§12 の前提 | [next_actions.md §8](next_actions.md) |
| B-2 | PPO 割当方策の保存機構 | `PPO` クラス側は完了。残りは `PPOAgent` の保存メソッド・yaml 5 キー・test.py の学習経路 | [next_actions.md §3](next_actions.md) |
| B-3 | PPO 学習診断 + TensorBoard | A-2 を含む | [next_actions.md §4](next_actions.md) |
| B-4 | 学習時の台数汎化 / 動的台数制御 / 同時学習 | いずれも未着手。動的台数制御は変更面積が最大 (env の 8 箇所) | [next_actions.md §9 / §10 / §12](next_actions.md) |

### 7.3 実験設計の衝突 (判断が必要・コードでは解けない)

- **C-1. 同時学習と「経路方策を全条件で同一にする」統制の衝突**: [aamas_submission.md §5.6.2](aamas_submission.md) は ①〜④ で同一の MAT を使う前提だが、[next_actions.md §12](next_actions.md) の同時学習では ③④ の経路方策が別物になる。結果 **③ − ① が「割当学習の効果 + 経路方策の共適応」の混合**になる。階層学習のアブレーション (凍結 vs 同時) を立てるかどうかとセットで判断する (2026-08-12 時点で保留)
- **C-2. 複数 seed 要件は割当方策側だけが未対応** (2026-08-13 に範囲を限定): [aamas_submission.md §5.4](aamas_submission.md) が複数 seed を要求している。**経路方策は sacred の自動生成で既に満たしている** (run ごとに別 seed・値はディレクトリ名に記録)。残るのは**割当方策 (PPO) 側だけ**で、A-3 のとおり torch が未シード (再現不可) かつ np が固定値 0 (replicate が同じタスク列を見る) のため、A-3 を直して `seed` を run ごとに変える運用が要る
- **C-3. PPO 割当方策が N 依存**: 入出力次元が `agent_num * node_num * 2 + task_num * node_num + agent_num` / `task_num * agent_num` で台数に依存するため、条件 ③ は **N ごとに別モデルの学習が必要**。設計には織り込み済み ([aamas_submission.md §5.1](aamas_submission.md) の「PPO は N 依存」) だが工数として残る。マスクでは解決できない (マスクは固定サイズの行動空間内で合法性を決めるだけ)

### 7.4 前提条件の欠落

- **D-3. タスク到着設定が test.py から env へ届いていない** (2026-08-14 発見・**実験の前提を壊す**): [test.py](../test.py) の `lare_path_keys` が転送するのは `randomize_task_arrival` だけで、**`task_arrival` / `task_density` / `task_p_high` / `task_p_low` / `task_switch_prob` を転送していない**。[src/config/default.yaml](../src/config/default.yaml) にもこれらのキーが無い。結果、env のシグネチャ既定値 [`task_arrival="fixed"`](../src/main/drp_env/drp_env.py) = **毎ステップ必ず 1 件到着**で走っている。
  - **実測の影響**: `task_num=10` のキューが即座に飽和する。スモークで「到着 14.0 件 / 取りこぼし 36.0 件 (72%) / 未ピック在庫ピーク 10.00 (上限張り付き)」。**`task_completion` が方策の質ではなく環境のスループット上限で頭打ちになる**ため、割当方策を学習しても信号が出ない (実際に PPO 学習中の `env/task_completion` が全区間 0.00 だった)。
  - [aamas_submission.md §5.6.1](aamas_submission.md) が実測で決めた bernoulli / mmpp のパラメータが**評価にも PPO 学習にも一切効いていない**ことになる。
  - **対処**: `lare_path_keys` に 5 キーを追加し、default.yaml に §5.6.1 の値を書く。あわせて飽和検知のため `saturated_ratio` / `task_dropped` を TensorBoard の `env/*` に足すと、「方策が悪い」のか「環境が飽和している」のかを切り分けられる。


- **D-1. station ノードが足りない**: 動的台数制御 ([dynamic_agent_count.md](dynamic_agent_count.md)) の帰投先。定義済みは map_3x3 (node 4) / map_5x4 (node 7) / map_10x8 (node 35) / map_aoba00 (node 2) / map_aoba01 (node 8) のみで、**実験でよく使う map_8x5 には無い**。しかもどれも 1 マップ 1 個なので、投入・退避が 1 拠点に集中する懸念もある。node.csv の 5 列目を 1 にすれば追加できる (旧 3 列形式のマップは列の追加も必要)
- **D-2. 経路方策の系譜が辿れない**: [CLAUDE.md](../CLAUDE.md) Step 2 の手動コピーでファイル名が平坦化され、`src/all_policy/models/safe/*.th` から「どの run 由来か」が失われる。`.th` 以外のファイルも置かれていないため復元手段が無い。コピーを自動化して系譜 json を同時に書くのが確実

### 7.5 ドキュメントのずれ

- **E-1. [CLAUDE.md](../CLAUDE.md) の学習出力先が古い**: `src/epymarl/tmp_results/models/{N}_{map}_safe_{algo}/` と書かれているが、実際の最近の学習は `results/models/{unique_token}/{step}/` に出ている (`tmp_results/` 側には旧形式のものが残っているだけ)
- **E-2. 保存命名規則の適用範囲が未記載**: CLAUDE.md の `{Safe_}{ALGO}_{PATH|TASK}_...pth` は **LaRe 限定**の規約だが、割当方策にも適用されるように読める。割当方策は epymarl の方策モデルと同じ規約に揃える方針 ([next_actions.md §3-1](next_actions.md))
- **E-3. [aamas_submission.md](aamas_submission.md) §2.4 / §5.2 が未修正**: ②-a (ピック前タスク再割当) を「必須前提」と書いたままで、[next_actions.md §11](next_actions.md) の決定 (station 方式により不要化) が反映されていない
- **E-4. 書誌の未確定**: Agarwal et al. と MaskMA は preprint、TransfQMix / PMAT のページ番号が未確定 ([aamas_submission.md §10.3](aamas_submission.md))

---

最終更新: 2026-08-03 (§6 エージェント稼働状態の指標化を追加)
最終更新: 2026-08-12 (§7 既知の問題インデックスを追加。2026-08-12 時点の未解決事項 A-1〜E-4 を棚卸し)

---

## 8. 学習スイープと実験管理 GUI

本文は [run_collector.md §12](run_collector.md#12-将来の実装) に置いてある。要点だけ:

- **学習スイープ** (`tools/train_sweep.py`): 評価 ([run.py](../run.py)) は条件 × seed を自動で回せるのに、
  **学習 ([train.py](../train.py)) だけ手動**。条件を変えるたびに f-string を書き換えて起動し直している。
  1 run が 5〜17 時間かかるので「条件 A の 5 seed が終わったら条件 B」を無人で回したい。
  seed を明示指定する (現状は sacred 任せで、Notion に書いた計画 seed と一致しない) のが前提。
- **実験管理 GUI** (`tools/exp_server.py`): ローカル Web サーバに決定 (2026-08-29)。閲覧だけでなく
  **この Mac から全マシンの学習を起動・予約できる管制画面**にする。
  (1) 複数マシンの監視 / (2) この PC から全マシンで起動 / (3) 条件ごとの完了 seed 一覧 /
  (4) 足りない seed をその場で実行 / (5) **空きメモリができたら自動起動する予約キュー**。
  段階は「監視だけ → 起動 → 予約」の順。
- **train.py の実行予定を枠として見せる**: `train.py` に数行足して `~/.ldrp/batch_<pid>.json` に
  予定本数を書き出す。収集側は既に読めるので、入れた瞬間に「白 1 run +4 wait」と出るようになる。
  本文は [run_collector.md §12.3](run_collector.md#123-trainpy-の実行予定を枠として見せる)。

---

## 9. 非アクティブ機の観測と衝突判定の不整合

`use_dynamic_agents=True` で off にしたエージェントの扱いが、**衝突判定と観測で食い違っている**。

| 側面 | 非アクティブ機の扱い | 実装 |
|---|---|---|
| 経路の行動空間 | 現在ノードに留まる 1 手だけ avail | [`_get_avail_agent_actions`](../src/main/drp_env/drp_env.py#L697) |
| タスク割当の行動空間 | そのエージェントのスロットを丸ごとマスク | [ppo.py](../src/task_assign/task_policy/ppo.py) の `mask` |
| 衝突判定 | **除外** (マップ上にいない扱い) | `collision_detect(..., active=...)` / `_lare_compute_colliding_pairs` |
| 報酬 | 0 (`start_ori_array[i] = goal_array[i]` にされる) | `reward()` |
| **観測 (FOV)** | **除外されていない** | [fov_wrapper.py](../src/main/drp_env/state_repre/wrapper/fov_wrapper.py) に `active` の参照が無い |

`calc_neighbor_filter` は「そのノードに他機がいれば -1」を立てるので、**ステーションに停まっている off 機は他機から障害物として見えたまま**になる。結果として off 機は

- 衝突判定からは除外される (ハード障害物ではない)
- 観測では占有として見える (ソフト障害物として避けられる)

という半端な状態になる。`exclude_station_from_tasks` が動的時は既定 True でステーションが pickup/dropoff 対象から外れるため実害は小さいが、**経路がステーションノードを通る地形では他機が不要な迂回をする**。

**どちらに寄せるかは物理的な解釈次第**なので、先に決める必要がある。

| 解釈 | 直すべき側 | 変更内容 |
|---|---|---|
| off = 機体がマップから消える | **観測** | `fov_wrapper` に `active` を渡し、非アクティブ機を占有から外す |
| off = ステーションに停まっているだけ | **衝突判定** | `collision_detect` の `active` 除外をやめ、停車中も衝突対象にする |

前者のほうが「稼働台数を減らす = 輻輳が減る」という [dynamic_agent_count.md](dynamic_agent_count.md) の狙いと整合する。

マスク方式そのものの性能上の代償と代替案は
[dynamic_agent_count.md §3 マスク方式の実装状況と性能上の代償](dynamic_agent_count.md) を参照。

---

## 10. 検出済みの学習条件の食い違い

`tools/collect_runs.py` の `param_hash` (seed とパスを除いた config の 8 桁ハッシュ) で
2026-09-07 に検出したもの。白ローカルの 70 条件のうち **2 条件**でパラメータが割れていた。

```bash
python tools/collect_runs.py -c tools/collect_config.yaml --format summary
# ... [warn] params differ across seeds: 53009eb4(1)  7095a604(1)
```

### (a) LaRe エンコーダの学習 on/off が seed 間で違う ← 要対応

```text
条件: 7agent map_aoba00 5M | qmix | 8x5_7 20M | bernoulli, mmpp | PPO
  seed 740597639 : env.use_lare_path_training  未指定 (= False)  → エンコーダ凍結
  seed 594911358 : env.use_lare_path_training  True             → エンコーダも学習
```

**同じ条件として平均できない。** `use_lare_path_training` は LaRe-Path の報酬エンコーダを
オンライン学習するかどうかのフラグで、凍結と学習では別手法になる。

対応: どちらを本条件にするか決めて、もう一方を捨てて再実行する。
`train.py` は現在 `use_lare_path_training=True` なので、そのまま回すと True 側に揃う。

### (b) `env_args.t_max` に値が渡されている ← 実害なし

```text
条件: 7agent map_aoba00 150M | mat_dec | 8x5_2 10M aoba00_2 5M
  seed 210715797 : env.t_max = 80050000 が付いている
```

`t_max` は epymarl の **top-level** の config キーで、`env_args` に渡しても env 側は受け取らない
(`DrpEnv.__init__` に `t_max` は無い)。実際の学習は top-level の 150.05M で走っているので
結果に影響はない。起動コマンドの書き間違いが残っているだけ。
なおこの条件の 3 本はいずれも `FAIL` なので、そもそも使えない run。

### 検出の仕組みについて

`param_hash` は 3 段で正規化している。これを入れないと誤検出が出る (実測)。

| 正規化 | 入れないとどうなるか |
|---|---|
| `None` / `False` はキー無しと同一視 | config のスキーマが版ごとに増えているため、**20 条件が誤検出**された |
| 有効化フラグが off のとき無視されるキーを落とす | `use_lare_path=False` の run に残っている `use_finetuning_lare_path=True` で誤検出 |
| env 側でハードコードされたキーを無視 (`cfg.task_num`) | env が `self.task_num = 10` を固定で持つので、config に出ていても挙動を変えない |

`param_fields()` が正規化を 1 箇所に持ち、`param_hash` と差分表示の両方がそれを通る。
別々に正規化すると「ハッシュは同じなのに差分が出る」食い違いが起きる (実際に一度起きた)。
