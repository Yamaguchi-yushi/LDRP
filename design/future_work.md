# 将来実装メモ (TODO 集約)

軽量な「将来やりたい」「未適用の修正」を集約するファイル。重い独立設計書 (例: [ldrp_extensions.md](ldrp_extensions.md)) はここから参照のみ。

各項目は **背景 / 現状 / 対策案 / 影響範囲** の節構成で書く。実装に着手したら本ファイルから対応セクションを削除し、必要に応じて [../MANUAL.md](../MANUAL.md) の更新履歴に記録する。

---

## 目次

1. [LaRe-Path 因子の正規化 (3 因子)](#1-lare-path-因子の正規化-3-因子)
2. [LaRe-Path 距離因子の残課題 (エッジ補間精度・タスク切替時の prog_goal)](#2-lare-path-距離因子の残課題-エッジ補間精度タスク切替時の-prog_goal)
3. [MAT-Dec 学習済みモデルの評価実行 (test.py) 対応](#3-mat-dec-学習済みモデルの評価実行-testpy-対応)

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

最終更新: 2026-07-16
