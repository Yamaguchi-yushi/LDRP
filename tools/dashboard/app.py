#!/usr/bin/env python3
"""LDRP 実験ダッシュボード (make_graph と同じ Flask + templates/ + static/ 構成).

    python tools/dashboard/app.py
    -> http://127.0.0.1:8765

タブ:
  学習   各マシンの run の条件と進捗・終了予定・seed 充足状況
  評価   results/summary.csv の集計結果

設計の要点:

  - **GET は待たせない**。全ホストへの ssh は 1〜2 分かかる (GPU2 は load 34 まで
    上がる) ので、画面表示はキャッシュ (tools/.run_cache.jsonl) を読むだけにする。
    収集は POST /api/collect か launchd に任せる。
  - 集計・状態判定・条件の分解は collect_runs.py / eval_report.py を import して
    使う。ここには**再実装しない**。
  - flask があれば flask、無ければ標準ライブラリの http.server で動く。
    ルーティングとハンドラは共通なので、どちらでも同じ画面になる。
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
REPO = os.path.dirname(TOOLS)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 表示順。ベースライン -> 提案 の順に並ぶようにする
ALGO_ORDER = ("qmix", "mappo", "mat", "mat_dec", "qplex", "iql", "vdn", "transf_qmix")

# 計画の枠 (5 seed) を埋められる run の状態。failed / stalled / short は
# 「回し直しが要る」ので枠を占めさせない (占めると未実行の数が実態より減る)
SLOT_STATES = ("done", "running")

# 「要確認」に出す状態。collect_runs.py の通知と同じ集合にしておく
# (通知で知らせたものが画面にも残る = 見落としの経路を 1 本にする)
ATTENTION_STATES = ("failed", "stalled", "short")

# OK を押した run を覚えておくファイル。tools/.host_seen.json と同じ置き方
ACK_PATH = os.path.join(TOOLS, ".acked_runs.json")

# 手で「5 seed に数えない」にした run。{uid: 除外した時刻}。
# 報酬の計算式を変えたときなど、config に残らない違いで使えなくなった run を
# 本数・多数決・設定の差分・全体比較のすべてから外す (表示は残して取り消せる)
EXCLUDE_PATH = os.path.join(TOOLS, ".run_excluded.json")
# 除外した run の評価用モデルに付ける印。run.py の model_seeds="auto" は
# <stem>_seed{K}.th にしか一致しないので、これを付ければ評価の対象から外れる
EXCLUDED_SUFFIX = ".excluded"

# seed ごとの手書きメモ。{uid: "本文"}。run が消えない限り残す
NOTE_PATH = os.path.join(TOOLS, ".run_notes.json")
# マシンごとの稼働率 (CPU / メモリ / GPU) の最新値。ダッシュボードを再起動しても
# 直前の値を出せるようにファイルにも置く (中身は collect_runs.host_stats の戻り値)
HOST_STATS_PATH = os.path.join(TOOLS, ".host_stats.json")
NOTE_MAX = 300      # 表の 1 列に収まる長さ。超える分は切る

# --- 全体比較 (条件をまたいで設定を比べる) ---------------------------------
# 条件内の多数決 (collect_runs.odd_param_runs) は、ある条件の 5 seed が**揃って**
# 古い設定だと何も言えない。表に載っている run 同士を条件をまたいで比べて、
# その穴を埋める。
#
# 比べないキー = 条件そのものを決めるもの (条件ごとに違って当然):
#   map / 台数 (env.key), t_max, algo (cfg.name), seed, LaRe 設定 (setting 列),
#   タスク到着の方式 (task arrival 列), 割当学習 (task assign 列), 動的台数 (dynamic 列),
#   学習時の再割当 (reassign 列。plan_reassign.md で条件として振っている)
GLOBAL_SKIP_KEYS = frozenset((
    "env.key", "cfg.t_max", "cfg.name", "cfg.env_args", "cfg.seed", "env.seed",
    "env.task_arrival", "env.randomize_task_arrival",
    "env.use_lare_path", "env.use_lare_path_training",
    "env.use_pretrained_lare_path", "env.pretrained_lare_path_model_name",
    "env.use_finetuning_lare_path", "env.finetuning_lare_path_model_name",
    "cfg.train_task_assigner", "env.use_dynamic_agents",
    "env.allow_reassign_before_pickup",
    # 学習時のタスク発生率の範囲。計画表の task arrival 列 (p=下限-上限) で
    # 条件として振っている (plan_arrival_range.md)。MMPP の 2 相も範囲の両端にそろえる。
    # 条件の中で割れたときの params✗ は今までどおり出る
    "env.rand_p_min", "env.rand_p_max", "env.task_p_low", "env.task_p_high",
))
GLOBAL_MIN_GROUP = 3    # 比べる相手がこれ未満なら多数派を決めない
GLOBAL_MAJORITY = 0.5   # 多数派とみなす割合 (これ未満 = 意図的に振っているキー)
# 「この差は問題ない」と判断したものを覚えておくファイル
DISMISS_PATH = os.path.join(TOOLS, ".param_dismissed.json")


def _norm(v):
    return json.dumps(v, sort_keys=True, default=str)


def _raw_param(d, k):
    """表示用に生の値を引く.

    param_fields は False / None を「キーが無い」に潰す (ハッシュを安定させるため)。
    判定はそちらに合わせるが、表示で null と出すと use_rnn=False が「記録なし」に
    見えてしまうので、画面には config に書かれていた値そのものを出す。
    """
    pre, key = k.split(".", 1)
    src = (d.get("cfg") or {}) if pre == "cfg" else (d.get("env") or {})
    # 報酬を渡していない run は drp_env の既定値で学習している (「記録なし」ではない)
    if k == "env.reward_list" and src.get(key) is None:
        return CR.DEFAULT_REWARD_LIST
    return src.get(key)


def _pdiff_val(key, v):
    """条件内の差分表に出す値. "-" は画面で「キー無し (古い版)」と出る."""
    if v is None:
        # 報酬は既定値のとき param_fields がキーを落とす。古い版ではなく既定値
        return str(CR.DEFAULT_REWARD_LIST) if key == "env.reward_list" else "-"
    return str(v)


def dismiss_key(cond, key, actual):
    """確認済みにする鍵. 値まで入れておけば、別の値に変わったとき再び出る."""
    return "%s|%s|%s" % (cond, key, _norm(actual))


def global_param_diff(rows):
    """表に載っている run 同士で、条件をまたいで設定を比べる.

    **同じ algo・同じ dynamic 設定の run はハイパーパラメータと env 設定が揃って
    いるはず**、という前提でキーごとに多数派の値を決め、そこから外れている条件を返す。
    algo をまたがないのは、mappo と qmix で lr が違うのは設計どおりだから。
    dynamic で分けるのは、動的台数の run にしか無いキー (min_active_agents など) が
    あるから。

    返り値: {condition: [{key, expected, actual, n_this, n_major, n_group}, ...]}
    """
    from collections import Counter, defaultdict
    groups = defaultdict(list)
    for d in rows:
        if d.get("state") in ("done", "running") and d.get("cfg"):
            groups[(d.get("algo"), bool(d.get("dynamic_agents")))].append(d)
    out = defaultdict(list)
    for ds in groups.values():
        if len(ds) < GLOBAL_MIN_GROUP:
            continue
        fields = [CR.param_fields(d.get("cfg"), d.get("env")) for d in ds]
        keys = set().union(*fields) - GLOBAL_SKIP_KEYS - set(CR.HASH_IGNORE_FIELDS)
        for k in sorted(keys):
            vals = [_norm(f.get(k)) for f in fields]
            count = Counter(vals)
            if len(count) < 2:
                continue
            major, n_major = count.most_common(1)[0]
            if n_major < len(ds) * GLOBAL_MAJORITY:
                continue
            ref = ds[vals.index(major)]
            off = defaultdict(list)
            for d, v in zip(ds, vals):
                if v != major:
                    off[(d["condition"], v)].append(d)
            for (cond, _v), xs in off.items():
                out[cond].append({
                    "key": k,
                    "expected": _raw_param(ref, k),
                    "actual": _raw_param(xs[0], k),
                    "n_this": len(xs), "n_major": n_major, "n_group": len(ds),
                })
    return out


def ack_key(r):
    """確認済みかどうかを見分ける鍵.

    uid だけにすると、同じ run を回し直して**また落ちた**ときに前回の OK が
    効いたままになり、二度目の失敗に気づけない。停止時刻まで入れておけば
    新しい失敗は別物として必ず出てくる。
    """
    return "%s|%s|%s" % (r.get("uid"), r.get("state"),
                         r.get("stop_at") or r.get("last_seen") or "")


def load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return {}


def save_json(obj, path):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except (IOError, OSError) as e:
        sys.stderr.write("[warn] could not save %s: %s\n" % (path, e))


def load_acks(path=ACK_PATH):
    return load_json(path)


def save_acks(acks, path=ACK_PATH):
    save_json(acks, path)


def _rank(seq, v):
    try:
        return seq.index(v)
    except ValueError:
        return len(seq)


CR = _load("collect_runs")
ER = _load("eval_report")
PLAN = _load("plan")


# ---------------------------------------------------------------------------
# データ
# ---------------------------------------------------------------------------

class State(object):
    """収集結果を持ち回す。GET はここを読むだけにする."""

    def _auto_collect_loop(self):
        """--collect-every 分ごとに裏で収集する.

        画面の再描画 (60 秒) はキャッシュを読み直すだけなので、これが無いと
        開きっぱなしでも中身は起動時のまま古くなる。
        """
        n = max(1, int(self.args.collect_every))
        while True:
            time.sleep(n * 60)
            if not self.busy:
                self.collect()

    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()
        self.busy = False
        self.collected_at = None
        self.errors = []
        self.batches = []          # train.py のバッチ。キャッシュには入れない
        # 共有フォルダの runs.jsonl の mtime。変わっていなければ読み飛ばす
        self.drop_mtime = {}

    # --- 設定 -----------------------------------------------------------
    def conf(self):
        path = os.path.expanduser(self.args.config)
        conf = CR.load_config(path) if os.path.exists(path) else {}
        CR.set_eval_naming(conf.get("eval_naming"))    # 評価用モデル名の形式
        return conf

    # --- 計画 -----------------------------------------------------------
    def plan_files(self):
        """読む計画ファイルの一覧.

        --plan が明示されていればそれ、無ければ collect_config.yaml の plans:、
        それも無ければ tools/plans/*.md → tools/plan.md の順に探す。
        """
        spec = self.args.plan
        if not spec:
            conf = CR.load_config(os.path.expanduser(self.args.config)) or {}
            spec = conf.get("plans")
        return PLAN.find_plans(spec)

    def pending_by_cond(self, conds):
        """train.py がこれから回す予定を、実験計画の条件ごとに数える.

        sacred は run が **始まって初めて** ディレクトリを作るので、5 本連続実行の
        2 本目を回している時点では 1 本しか見えない。マシンが埋まっているのに
        表は「未実行 4」のままになり、二重に投入してしまう。
        train.py が書いた予約 (`~/.ldrp/batch_<pid>.json` の cmd) を条件に
        対応づけて、残り (total - started) を「実行待ち」として枠に出す。

        条件の判定は実績の run と **同じ derive() 1 か所** で行う。
        コマンド行のキー名は config.json と同じなので、そのまま通せる。
        """
        out = {}
        conf = self.conf()
        for b in (self.batches or []):
            total, started = b.get("total"), b.get("started")
            cmd = b.get("train_cmd")
            if not cmd or total is None or started is None:
                continue                  # 予約を書いていない = 本数が分からない
            left = int(total) - int(started)
            if left <= 0:
                continue
            rec = CR.parse_train_command(cmd)
            rec.update({"uid": b.get("uid"), "machine": b.get("machine")})
            try:
                d = CR.derive(rec, conf.get("stale_minutes") or 90,
                              conf.get("method_tag_by_lare_mode"),
                              conf.get("expected_t_max"),
                              lare_chain=conf.get("lare_chain"))
            except Exception:             # 解析できない予約で画面を壊さない
                continue
            for i, c in enumerate(conds):
                if PLAN.matches(c, d):
                    e = out.setdefault(i, {"n": 0, "machines": []})
                    e["n"] += left
                    if b.get("machine") not in e["machines"]:
                        e["machines"].append(b.get("machine"))
                    break
        return out

    def plan_view(self, rows, shaped, excl=None):
        """**表は計画から作る**。実績はそこに埋めていく.

        計画に無い run は conditions 表には出さない (実行中セクションには出る)。
        こうすると「あと何を回せばよいか」が表そのものになる。

        slots に入れる run は **dashboard_data() が整形したもの** (shaped)。
        derive() の生レコードを入れると odd_params などが欠けて画面側が壊れる。
        """
        paths = self.plan_files()
        if not paths:
            return [], set()
        try:
            conds = PLAN.parse_plans(paths)
        except Exception as e:
            sys.stderr.write("[plan] %s: %s\n" % (type(e).__name__, e))
            return [], set()

        excl = excl or {}
        used = set()
        out = []
        # パラメータが多数派とずれている run。印を付ける判断に使う
        odd_uids = set(d["uid"] for d in CR.odd_param_runs(rows))
        pending = self.pending_by_cond(conds)
        # seed -> その seed を表に明記している条件の添字。別の計画 (例: AAMAS) の表に
        # seed が書いてある run は、その計画のもの。1 seed だけ借りている計画
        # (例: plan_reassign.md) の枠の下に「余分な完了 run *」として並べない
        written = {}
        for ci, c in enumerate(conds):
            for sl in c["seeds"]:
                if sl.get("seed"):
                    written.setdefault(str(sl["seed"]), []).append(ci)
        for ci, c in enumerate(conds):
            hit = [d for d in rows if PLAN.matches(c, d)]
            by_seed = {}
            for d in hit:
                by_seed.setdefault(str(d.get("seed")), []).append(d)

            slots = []
            blanks = []                       # seed 欄が空のままの枠 (添字)
            for sl in c["seeds"]:
                sd = str(sl["seed"]) if sl["seed"] else None
                run = None
                if sd and by_seed.get(sd):
                    # 除外した run では枠を埋めない。同じ seed で回し直した run が
                    # あればそちらを入れ、無ければ枠は空 (= 未実行) のまま。
                    # 除外した run は by_seed に残るので、下の leftover で枠の下に出る
                    cands = by_seed[sd]
                    d = next((x for x in cands if x["uid"] not in excl), None)
                    if d is not None:
                        cands.remove(d)
                        used.add(d["uid"])
                        run = shaped.get(d["uid"])
                slots.append({"seed": sd, "machine": sl.get("machine"),
                              "reassign": sl.get("reassign"), "run": run})
                if sd is None:
                    blanks.append(len(slots) - 1)

            # 計画に seed が書かれていない run。**まず空いている枠を埋める**。
            # 枠と別に並べると 5 seed の条件が「未実行 5 行 + 実績 2 行」の 7 行に
            # なり、あと何本回せばよいのかが表から読めなくなる (実測で 84 条件中
            # 46 条件が seed 欄を空けたまま自動検出に任せている)。
            # 枠より多い分だけを下に足す。
            leftover = sorted((d for ds in by_seed.values() for d in ds),
                              key=lambda d: (str(d.get("state") not in SLOT_STATES),
                                             str(d.get("start_time") or ""),
                                             str(d.get("seed"))))
            # 印 (*) を付けるのは **不具合が疑わしいときだけ**。状態 / params✗ /
            # t_max は別の欄に既に出ているので、ここで黄色にするのは「表に 5 seed
            # 書いてあるのに、さらに別の seed が回っている」場合 (取り違えか二重実行)。
            full = sum(1 for sl in c["seeds"] if sl.get("seed")) >= c["want"]
            planned = set(str(sl["seed"]) for sl in c["seeds"] if sl.get("seed"))
            extra = []
            for d in leftover:
                used.add(d["uid"])
                if full and any(cj != ci and PLAN.matches(conds[cj], d)
                                for cj in written.get(str(d.get("seed")), [])):
                    continue              # 別の計画の表に載っている run
                gone = d["uid"] in excl
                row = {"seed": str(d.get("seed")), "machine": d.get("machine"),
                       "run": shaped.get(d["uid"]),
                       # 表に無い seed = 補足情報。除外した run は計画の seed のまま
                       # 下に来るので、実際に計画に無いかで判定する
                       "unplanned_seed": str(d.get("seed")) not in planned,
                       # 除外済みは「別の seed も完了している」の疑いに数えない
                       "suspect": bool(full and d.get("state") == "done" and not gone)}
                # 失敗・停止した run は枠を埋めない。埋めてしまうと 5 行のうち
                # 1 行が失敗で占められ、「あと何本回せばよいか」が読めなくなる。
                # 消さずに下へ出す (どの seed が落ちたかは見たいため)
                if blanks and d.get("state") in SLOT_STATES and not gone:
                    i = blanks.pop(0)
                    row["reassign"] = slots[i].get("reassign")
                    slots[i] = row
                else:
                    extra.append(row)

            # train.py がこれから回す分を「実行待ち」として置く。**run が入って
            # いない枠すべて**が対象 (seed が明記された枠も含む)。train.py は
            # seed を自分で引くので予約とその seed 番号は結び付かないが、
            # 埋めたいのは「この条件はあと何本要るか」なので枠の別は問わない。
            # 枠より多く予約されていても表は 5 行のまま。
            p = pending.get(ci)
            if p:
                empty = [i for i, s in enumerate(slots) if not s.get("run")]
                for i in empty[:p["n"]]:
                    slots[i]["pending"] = True
                    slots[i]["pending_on"] = "/".join(x for x in p["machines"] if x)
            # 同じ条件のはずなのにパラメータが割れていたら、**どのキーが違うか**を渡す。
            # ハッシュだけでは何を直せばよいか分からない
            pdiff, phashes = [], {}
            # 比べるのは **5 seed に数える run** (完了・実行中で、除外していないもの) だけ。
            # 失敗・停止した run は数えないので、その設定が違っていても直す必要がない
            # (odd_param_runs / 全体比較と同じ対象にそろえる)
            hit_live = [d for d in hit if d["uid"] not in excl
                        and d.get("state") in SLOT_STATES]
            if len(set(d.get("param_hash") for d in hit_live)) > 1:
                diffs, seeds = CR.param_diff(hit_live)
                order = sorted(seeds, key=lambda h: (-len(seeds[h]), str(h)))
                phashes = [{"hash": h, "seeds": sorted(seeds[h])} for h in order]
                pdiff = [{"key": k,
                          "vals": [_pdiff_val(k, v.get(h)) for h in order]}
                         for k, v in diffs]

            out.append({
                "param_diff": pdiff, "param_hashes": phashes,
                "plan": c.get("plan"), "plan_file": c.get("plan_file"),
                "label": PLAN.label(c), "map": c["map"], "agents": c["agents"],
                "t_max_m": c["t_max_m"], "setting": c["setting"],
                "algo": c["algo"], "task_arrival": c["task_arrival"],
                "task_assign": c["task_assign"], "reassign": c["reassign"],
                "dynamic": c.get("dynamic"),
                "want": c["want"], "slots": slots + extra,
            })
        # 見出しは マップ -> 台数 -> t_max。
        # 表の中は**列の左から順に** setting -> algorithm -> arrival -> assign -> dynamic。
        # 列内の並びは意味のある順にする (safe が基準、QMIX->MAPPO->MAT、TP->PPO、F->T)
        out.sort(key=lambda c: (
            str(c.get("plan") or ""),
            str(c["map"]), c["agents"], c["t_max_m"],
            0 if c["setting"] == "safe" else 1, str(c["setting"]),
            _rank(ALGO_ORDER, c["algo"]), str(c["algo"] or ""),
            str(c["task_arrival"] or ""),
            0 if not c["task_assign"] else 1, str(c["task_assign"] or ""),
            1 if c.get("reassign") else 0,      # 再割当なし (F) を先に
            1 if c.get("dynamic") else 0,
        ))
        return out, used

    # --- 学習 -----------------------------------------------------------
    def train(self):
        conf = self.conf()
        raw = CR.read_cache(os.path.expanduser(self.args.cache))
        raw = [r for r in raw if r.get("kind") not in ("batch", "host")]
        stale = conf.get("stale_minutes") or 90
        rows = [CR.derive(r, stale, conf.get("method_tag_by_lare_mode"),
                          conf.get("expected_t_max"),
                          lare_chain=conf.get("lare_chain"))
                for r in CR.dedupe(raw)]
        min_steps = conf.get("min_steps", 1e6)
        if min_steps:
            # 短い run (動作確認用の t_max=数万) は表を埋めるので既定で落とす。
            # ただし **走っている間は必ず見せる**。報酬設計を変えた直後の試し
            # 実行のように、計画に無く t_max も小さい run こそ進捗を見たい。
            # 終われば消えるが、そのときは計画外の件数として数えられる
            rows = [d for d in rows
                    if (d.get("t_max") or 0) >= min_steps
                    or d.get("state") == "running"]
        with self.lock:
            data = CR.dashboard_data(
                rows, self.batches, self.errors,
                saved=CR.load_saved_models(self.args.models_repo),
                cadence=CR.record_host_seen(rows, hosts=self.conf().get("hosts")))
            # 除外した run は**判定に一切使わない**。多数決に残すと、たとえば旧報酬の
            # 3 本を外しても、回し直した新しい 2 本のほうが少数派として params✗ になる。
            # odd_params は dashboard_data() が全 run で決めているので、ここで決め直す
            excl = load_json(EXCLUDE_PATH)
            live = [d for d in rows if d["uid"] not in excl]
            odd = set(d["uid"] for d in CR.odd_param_runs(live))
            for r in data["runs"]:
                r["excluded"] = r["uid"] in excl
                r["excluded_at"] = excl.get(r["uid"])
                r["odd_params"] = r["uid"] in odd
            shaped = dict((r["uid"], r) for r in data["runs"])
            plan, used = self.plan_view(rows, shaped, excl)
            # run ごとの「どの計画の枠に入っているか」(running now に出す)。
            # 1 つの run が複数の計画に数えられることがある (AAMAS と reassign 等)
            plans_of = {}
            for c in plan:
                for sl in c.get("slots") or []:
                    u = (sl.get("run") or {}).get("uid")
                    if u and c.get("plan") and c["plan"] not in plans_of.setdefault(u, []):
                        plans_of[u].append(c["plan"])
            for r in data["runs"]:
                r["in_plan"] = r["uid"] in used
                r["plans"] = plans_of.get(r["uid"], [])
            self.attach_global_diff(plan, [d for d in live if d["uid"] in used])
            data["plan"] = plan
            data["plan_files"] = self.plan_files()
            data["plan_file"] = ", ".join(data["plan_files"])
            data["collected_at"] = self.collected_at
            data["busy"] = self.busy
            data["cache"] = os.path.expanduser(self.args.cache)
            notes = load_json(NOTE_PATH)
            for r in data["runs"]:
                r["note"] = notes.get(r["uid"], "")
            data["attention"] = self.attention(data["runs"])
            data["host_stats"] = load_json(HOST_STATS_PATH)
        return data

    # --- 全体比較 -------------------------------------------------------
    def attach_global_diff(self, plan, rows):
        """計画の各条件に、全体と違う設定 (global_diff) を付ける.

        比べる相手は**表に載っている run だけ** (計画外の試し実行や古い探索を
        混ぜると多数派がぶれる)。確認済みにしたもの (DISMISS_PATH) は外し、
        件数だけ n_dismissed として残す (取り消せるように)。
        """
        gdiff = global_param_diff(rows)
        dismissed = load_json(DISMISS_PATH)
        for c in plan:
            conds = set(s["run"]["condition"] for s in (c.get("slots") or [])
                        if s.get("run") and s["run"].get("condition"))
            live, n_dis = [], 0
            for cond in sorted(conds):
                for x in gdiff.get(cond, []):
                    x = dict(x, cond=cond,
                             dismiss_key=dismiss_key(cond, x["key"], x["actual"]))
                    if x["dismiss_key"] in dismissed:
                        n_dis += 1
                    else:
                        live.append(x)
            c["global_diff"] = live
            c["global_dismissed"] = n_dis
            c["global_conds"] = sorted(conds)     # 取り消し用 (確認済みの鍵の頭)

    def dismiss_diff(self, body):
        """全体との差を「問題なし」にする. body: {"key": dismiss_key}.

        {"undo": true} を付けると取り消す (この条件の確認済みを全部戻す)。
        """
        body = body or {}
        dismissed = load_json(DISMISS_PATH)
        if body.get("undo"):
            conds = body.get("conds") or [body.get("cond", "")]
            prefixes = tuple("%s|" % c for c in conds if c)
            keys = [k for k in dismissed if prefixes and k.startswith(prefixes)]
            for k in keys:
                dismissed.pop(k, None)
            save_json(dismissed, DISMISS_PATH)
            return {"undone": len(keys)}
        k = body.get("key")
        if not k:
            return {"ok": False, "error": "key is required"}
        dismissed[k] = CR.now_utc().isoformat()
        save_json(dismissed, DISMISS_PATH)
        return {"ok": True}

    # --- 手動除外 -------------------------------------------------------
    def exclude(self, body):
        """run を「5 seed に数えない」にする / 戻す.

        body: {"uid": ..., "on": true}  (false で戻す)
        """
        body = body or {}
        uid = body.get("uid")
        if not uid:
            return {"ok": False, "error": "uid is required"}
        on = bool(body.get("on", True))
        excl = load_json(EXCLUDE_PATH)
        if on:
            excl[uid] = CR.now_utc().isoformat()
        else:
            excl.pop(uid, None)
        save_json(excl, EXCLUDE_PATH)
        moved, errors = self._move_eval_models(uid, exclude=on)
        return {"ok": True, "uid": uid, "excluded": uid in excl,
                "moved": moved, "move_errors": errors}

    def _eval_names(self, uid):
        """この run が評価用フォルダに置かれたときのファイル名 (保管リポジトリの manifest から)."""
        path = os.path.join(os.path.expanduser(self.args.models_repo), "manifest.jsonl")
        names = []
        try:
            with open(path, "r") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    if r.get("uid") == uid and r.get("eval_name"):
                        names.append(r["eval_name"])
        except (IOError, OSError):
            pass
        return names

    def _move_eval_models(self, uid, exclude):
        """評価用フォルダのモデルに .excluded を付けて評価の対象から外す / 戻す.

        消さずに改名するだけなので「戻す」で元どおりになる。経路方策と割当方策は
        **同じ名前で別フォルダ**に置かれている (collect_runs.install_models)。
        戻す先に同じ名前のファイルが既にある場合は上書きしない。
        """
        moved, errors = [], []
        for name in self._eval_names(uid):
            for kind, rel in sorted(CR.EVAL_MODEL_DIRS.items()):
                d = os.path.join(REPO, rel)
                live = os.path.join(d, name)
                off = live + EXCLUDED_SUFFIX
                src, dst = (live, off) if exclude else (off, live)
                if not os.path.exists(src):
                    continue
                if os.path.exists(dst):
                    # 除外で空いた番号は回し直した run に引き継がれる
                    # (collect_runs.load_seed_index)。その後で「戻す」と同じ名前が
                    # 2 つになるので、上書きせずに理由を知らせる
                    if not exclude:
                        msg = ("%s: この番号は回し直した run が使っているので、"
                               "モデルは戻していません" % name)
                        if msg not in errors:
                            errors.append(msg)
                    continue
                try:
                    os.replace(src, dst)
                    moved.append("%s/%s" % (kind, os.path.basename(dst)))
                except OSError as e:
                    errors.append("%s: %s" % (name, e))
        return moved, errors

    # --- seed ごとのメモ -------------------------------------------------
    def note(self, body):
        """conditions 表に手で書いたコメントを保存する.

        body: {"uid": ..., "text": ...}。空文字なら削除する
        (空のキーを残すとファイルが使い回すたびに膨らむだけ)
        """
        body = body or {}
        uid = body.get("uid")
        if not uid:
            return {"ok": False, "error": "uid is required"}
        text = str(body.get("text") or "").strip()[:NOTE_MAX]
        notes = load_json(NOTE_PATH)
        if text:
            notes[uid] = text
        else:
            notes.pop(uid, None)
        save_json(notes, NOTE_PATH)
        return {"ok": True, "uid": uid, "text": text}

    # --- 要確認リスト ---------------------------------------------------
    def attention(self, runs):
        """異常終了した run のうち、まだ OK を押していないものを返す.

        OK 済みでも run 自体は conditions / counts にそのまま残る。ここで
        消えるのは「気づいてほしい」という呼びかけだけ。
        """
        acks = load_acks()
        out = []
        for r in runs:
            if r.get("state") not in ATTENTION_STATES:
                continue
            key = ack_key(r)
            if key in acks:
                continue
            out.append(dict(r, ack_key=key))
        # 新しく落ちたものほど上。時刻が無いものは末尾へ
        out.sort(key=lambda r: r.get("stop_at") or r.get("last_seen") or "",
                 reverse=True)
        return out

    def ack(self, body):
        """OK が押された run を確認済みにする.

        body: {"keys": [ack_key, ...]}  /  {"all": true} で表示中の全部
        """
        body = body or {}
        keys = list(body.get("keys") or [])
        if body.get("all"):
            keys += [r["ack_key"] for r in self.train()["attention"]]
        if not keys:
            return {"acked": 0}
        acks = load_acks()
        now = CR.now_utc().isoformat()
        for k in keys:
            acks[k] = now
        save_acks(acks)
        return {"acked": len(keys)}

    # --- 評価 -----------------------------------------------------------
    def eval(self):
        path = os.path.expanduser(self.args.summary)
        if not os.path.exists(path):
            return {"source": path, "available": False, "metrics": [],
                    "conditions": [],
                    "error": "summary.csv がありません (aggregate.py --csv を実行してください)"}
        rows, metrics = ER.load_summary(path)
        return {"source": path, "available": True, "metrics": metrics,
                "conditions": sorted(rows, key=ER.sort_key)}

    # --- 収集 (重い。バックグラウンドで回す) -----------------------------
    def collect(self):
        with self.lock:
            if self.busy:
                return False
            self.busy = True
        threading.Thread(target=self._collect_worker, daemon=True).start()
        return True

    def _collect_worker(self):
        try:
            conf = self.conf()
            hosts = conf.get("hosts")
            if not hosts:
                # 設定が読めていないのに os.uname()[1] を使うと、白の run が
                # ホスト名ラベルで二重に溜まる (uid にマシン名が入るので
                # dedupe では消えない)。収集せずに知らせて止める
                msg = ("collect_config.yaml から hosts を読めませんでした"
                       " (PyYAML が無い python で起動していませんか)")
                sys.stderr.write("[collect] %s\n" % msg)
                with self.lock:
                    self.errors = [("config", msg)]
                return
            raw, errors = CR.collect(hosts, conf.get("tail_bytes") or 65536,
                                     self.args.ssh_timeout,
                                     drop_root=conf.get("drop_root"),
                                     skip_unchanged=self.drop_mtime)
            batches = [r for r in raw if r.get("kind") == "batch"]
            hstats = [r for r in raw if r.get("kind") == "host"]
            runs = [r for r in raw if r.get("kind") not in ("batch", "host")]
            if hstats:
                # 届かなかったマシンの値は前回のものを残す (いつの値かは "at" で分かる)
                hs = load_json(HOST_STATS_PATH)
                for h in hstats:
                    hs[h.get("machine")] = h
                try:
                    with open(HOST_STATS_PATH + ".tmp", "w") as f:
                        json.dump(hs, f, ensure_ascii=False)
                    os.replace(HOST_STATS_PATH + ".tmp", HOST_STATS_PATH)
                except (IOError, OSError) as e:
                    sys.stderr.write("[host] %s\n" % e)

            cache = os.path.expanduser(self.args.cache)
            merged = CR.dedupe(CR.read_cache(cache) + runs)
            stale = conf.get("stale_minutes") or 90
            rows = [CR.derive(r, stale, conf.get("method_tag_by_lare_mode"),
                              conf.get("expected_t_max"),
                              lare_chain=conf.get("lare_chain"))
                    for r in merged]
            by_uid = dict((d["uid"], d["state"]) for d in rows)
            for r in merged:
                r["_state"] = by_uid.get(r.get("uid"))
            CR.write_cache(cache, merged)

            with self.lock:
                self.batches = batches
                self.errors = ["%s: %s" % (a, b) for a, b in errors]
                self.collected_at = time.time()
        except Exception as e:                      # 画面を落とさない
            with self.lock:
                self.errors = ["collect failed: %s: %s" % (type(e).__name__, e)]
            sys.stderr.write("[collect] %s: %s\n" % (type(e).__name__, e))
        finally:
            with self.lock:
                self.busy = False

    def status(self):
        with self.lock:
            return {"busy": self.busy, "collected_at": self.collected_at,
                    "errors": self.errors}

    def mini(self):
        """デスクトップの小さいパネル用の要約 (1KB 未満).

        train() は 1.8MB あり、常駐パネルが毎分取りにくるには重い。
        中身は同じ計算なので、要るものだけ抜いて返す。
        """
        d = self.train()
        plan = d.get("plan") or []
        full = part = todo = 0
        need = running = 0
        for c in plan:
            want = c.get("want") or 5
            slots = (c.get("slots") or [])[:want]
            done = sum(1 for s in slots
                       if s.get("run") and s["run"].get("state") == "done")
            kept = sum(1 for s in slots
                       if s.get("run") and (s["run"].get("saved") or []))
            run = sum(1 for s in slots
                      if s.get("run") and s["run"].get("state") == "running")
            need += max(0, want - done - run)
            running += run
            if done >= want and kept >= want:
                full += 1
            elif done or run:
                part += 1
            else:
                todo += 1
        # マシンごとの「最後に何を終わらせたか」。実行中が 0 でも、直前に
        # 終わったのが 3 日前なら手が空いている = 投入すべき、と判断できる
        last = {}
        nxt = {}        # マシンごとに「次に終わる予定の実行中 run」
        paused = {}     # マシンごとの一時停止中 (Ctrl-Z) の run
        for r in d.get("runs") or []:
            k = r.get("machine")
            if r.get("state") == "running" and r.get("paused"):
                paused.setdefault(k, []).append(r)
            elif r.get("state") == "running" and r.get("eta"):
                if k not in nxt or r["eta"] < nxt[k]["eta"]:
                    nxt[k] = r
            if r.get("state") != "done" or not r.get("stop_at"):
                continue
            if k not in last or r["stop_at"] > last[k]["stop_at"]:
                last[k] = r
        def what(r):
            return "%sag %s %s" % (r.get("agents"), str(r.get("map") or "").replace("map_", ""),
                                   r.get("algo"))

        machines = []
        for name, m in sorted((d.get("machines") or {}).items()):
            r = last.get(name)
            e = nxt.get(name)
            machines.append({
                "name": name, "running": m.get("running") or 0,
                "age": m.get("data_age_sec"), "stale": bool(m.get("stale_data")),
                "last_done": r.get("stop_at") if r else None,
                "last_what": what(r) if r else None,
                # 次の終了予定。実測ペースから出した見込みなので、情報が古いホストでは当てにならない
                "next_eta": e.get("eta") if e else None,
                "next_what": what(e) if e else None,
                # 一時停止中の本数と、再開したときに最も早く終わるものの残り時間
                "paused": len(paused.get(name, [])),
                "paused_remaining": min((x.get("remaining_sec") for x in paused.get(name, [])
                                         if x.get("remaining_sec") is not None), default=None),
                "paused_what": what(paused[name][0]) if paused.get(name) else None,
            })
        return {"conditions": len(plan), "full": full, "part": part, "todo": todo,
                "need": need, "running": running, "machines": machines,
                "collected_at": self.collected_at, "busy": self.busy}


# ---------------------------------------------------------------------------
# ルーティング (flask / 標準ライブラリのどちらからも同じものを呼ぶ)
# ---------------------------------------------------------------------------

def make_routes(state):
    return {
        ("GET", "/api/train"): lambda body: state.train(),
        ("GET", "/api/eval"): lambda body: state.eval(),
        ("GET", "/api/status"): lambda body: state.status(),
        ("GET", "/api/mini"): lambda body: state.mini(),
        ("POST", "/api/collect"): lambda body: {"started": state.collect()},
        ("POST", "/api/ack"): lambda body: state.ack(body),
        ("POST", "/api/note"): lambda body: state.note(body),
        ("POST", "/api/dismiss_diff"): lambda body: state.dismiss_diff(body),
        ("POST", "/api/exclude"): lambda body: state.exclude(body),
    }


def serve_flask(state, host, port):
    from flask import Flask, jsonify, render_template, request
    app = Flask(__name__, template_folder=os.path.join(HERE, "templates"),
                static_folder=os.path.join(HERE, "static"))
    routes = make_routes(state)
    # stdlib 側と同じく、開発中は常に取り直させる
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

    @app.after_request
    def _nocache(resp):
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/<path:rest>", methods=["GET", "POST"])
    def api(rest):
        fn = routes.get((request.method, "/api/" + rest))
        if fn is None:
            return jsonify({"error": "not found"}), 404
        body = request.get_json(silent=True) or {}
        return jsonify(fn(body))

    sys.stderr.write("[dashboard] flask  http://%s:%d\n" % (host, port))
    app.run(host=host, port=port, threaded=True, debug=False)


def serve_stdlib(state, host, port):
    """flask が無いとき用。templates/index.html に Jinja 構文が無いのでそのまま返せる."""
    import mimetypes
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    routes = make_routes(state)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, data, ctype):
            body = data if isinstance(data, bytes) else data.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            # 開発中のツールなので **常に取り直させる**。
            # Cache-Control も ETag も付けないと、ブラウザが発見的に
            # app.js を握り続け、直したはずの画面が変わらない
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(body)

        def do_HEAD(self):
            # 501 Unsupported method ('HEAD') を返さないようにする (curl -I 用)
            self.do_GET()

        def _file(self, path, ctype=None):
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                return self._send(404, b"not found", "text/plain")
            self._send(200, data, ctype or
                       (mimetypes.guess_type(path)[0] or "application/octet-stream"))

        def _api(self, method):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                body = {}
            fn = routes.get((method, self.path.split("?")[0]))
            if fn is None:
                return self._send(404, json.dumps({"error": "not found"}),
                                  "application/json")
            self._send(200, json.dumps(fn(body), ensure_ascii=False, default=str),
                       "application/json; charset=utf-8")

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html"):
                return self._file(os.path.join(HERE, "templates", "index.html"),
                                  "text/html; charset=utf-8")
            if p.startswith("/static/"):
                rel = os.path.normpath(p[len("/static/"):]).lstrip(os.sep)
                return self._file(os.path.join(HERE, "static", rel))
            if p.startswith("/api/"):
                return self._api("GET")
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/api/"):
                return self._api("POST")
            self._send(404, b"not found", "text/plain")

    try:
        srv = ThreadingHTTPServer((host, port), H)
    except OSError as e:
        return port_in_use(host, port, e)
    sys.stderr.write("[dashboard] stdlib http.server  http://%s:%d\n"
                     "[dashboard] (pip install flask すると flask で動きます)\n"
                     % (host, port))
    srv.serve_forever()


def port_in_use(host, port, err):
    """ポートが埋まっているときに **何をすればよいか** を出す.

    素の OSError だと Flask の ImportError を巻き込んだ二重トレースバックになり、
    「flask が無いのが原因」と誤読しやすい。原因と対処だけを出す。
    """
    import errno
    if getattr(err, "errno", None) != errno.EADDRINUSE:
        raise err
    sys.stderr.write("[dashboard] port %d on %s is already in use.\n" % (port, host))
    try:
        out = subprocess.check_output(
            ["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN"],
            stderr=subprocess.DEVNULL).decode("utf-8", "replace").splitlines()
        for line in out[1:]:
            f = line.split()
            if len(f) > 1:
                sys.stderr.write("[dashboard]   pid %s (%s) is holding it\n"
                                 % (f[1], f[0]))
    except (OSError, subprocess.CalledProcessError):
        pass
    sys.stderr.write("[dashboard] either open http://%s:%d (it is probably the "
                     "same dashboard), stop that pid, or pass --port\n"
                     % (host, port))
    return 1


def main(argv=None):
    p = argparse.ArgumentParser(description="LDRP experiment dashboard.")
    p.add_argument("-c", "--config", default=os.path.join(TOOLS, "collect_config.yaml"))
    p.add_argument("--cache", default=os.path.join(TOOLS, ".run_cache.jsonl"),
                   help="collect_runs.py --cache と同じファイル")
    p.add_argument("-p", "--plan", action="append", default=None,
                   help="実験計画 (Notion の表をそのまま貼れる). glob 可。"
                        "複数指定でき、省略時は tools/plans/*.md -> tools/plan.md")
    p.add_argument("--models-repo", default="~/LDRP_models",
                   help="方策モデルの保管リポジトリ。manifest.jsonl を読んで "
                        "「回収済みか」を表示する")
    p.add_argument("-s", "--summary", default=os.path.join(REPO, "results", "summary.csv"))
    p.add_argument("--host", default="127.0.0.1", help="既定は localhost のみ")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--ssh-timeout", type=int, default=180)
    p.add_argument("--collect-on-start", action="store_true",
                   help="起動時に 1 回収集する (既定はキャッシュを読むだけ)")
    p.add_argument("--collect-every", type=int, default=15, metavar="MIN",
                   help="この分数ごとに裏で収集する (0 で無効。既定 15)。"
                        "共有フォルダ側が 1 時間おきにしか更新されないので、"
                        "これより短くしても情報は増えない")
    args = p.parse_args(argv)

    state = State(args)
    if args.collect_on_start:
        state.collect()
    if args.collect_every:
        threading.Thread(target=state._auto_collect_loop, daemon=True).start()
        sys.stderr.write("[dashboard] auto collect every %d min\n" % args.collect_every)

    try:
        import flask                                     # noqa: F401
    except ImportError:
        return serve_stdlib(state, args.host, args.port)
    return serve_flask(state, args.host, args.port)


if __name__ == "__main__":
    sys.exit(main())
