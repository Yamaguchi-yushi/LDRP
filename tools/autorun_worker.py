#!/usr/bin/env python3
"""ssh で届かないマシン (黒など) で、司令塔 (tools/autorun.py) が共有フォルダに置いたジョブを起動する受け取り役.

launchd から 5 分おきに 1 回ずつ呼ぶ (登録: python tools/autorun_worker.py --install)。1 回の実行でやること:
  1. <drop_root>/<label>/autorun/inbox/*.json を started/ に移して起動する
       train: 添付の LaRe モデルを train_repo に置き、autorun_train.py で epymarl を起動 (Mac ではスリープ防止つき)
       eval : コードの束を eval_repo に展開し (中身が変わったときだけ)、方策モデルを置いて
              tools/eval_lare_factor.py を CPU で起動
  2. eval_repo/logs/ の *.txt を共有フォルダの eval_logs/ に写す (司令塔の collect が手元の logs/ に取り込む)
  3. このマシンの空き具合などを status.json に書く (司令塔はこれを ssh の代わりに読む)

設定は tools/collect_config.yaml (このマシンのもの):
  machine: 黒                       # このマシンのラベル (司令塔の collect_config.yaml の label と同じ)
  drop_root: ~/Library/Mobile Documents/com~apple~CloudDocs/LDRP_runs
  autorun_commander: 白             # 司令塔のラベル (これがあると autorun.py の add などが司令塔に送られる)
  autorun_worker:                   # 省略可。省略時は下の既定値
    python: /opt/anaconda3/envs/ldrp/bin/python   # 既定: この worker を動かしている Python
    train_repo: ~/LDRP                            # 既定: この worker があるリポジトリ
    eval_repo: ~/LDRP_eval                        # 既定: ~/LDRP_eval (評価用のコードの写しを置く場所)

使い方の全体: tools/AUTORUN.md
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
sys.path.insert(0, TOOLS)
import autorun as AR        # noqa: E402  (PROBE / EVAL_CODE などを司令塔と揃える)

HOME_DIR = os.path.expanduser("~/" + AR.REMOTE_DIR)       # ~/.ldrp/autorun (ジョブ JSON・ログ)
ERRORS = os.path.join(HOME_DIR, "worker_errors.json")


def log(msg):
    print("[worker %s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg), flush=True)


def settings():
    conf = AR.load_conf()
    label, root = conf.get("machine"), conf.get("drop_root")
    if not label or not root:
        raise SystemExit("set `machine:` and `drop_root:` in %s" % AR.CONFIG)
    w = conf.get("autorun_worker") or {}
    ex = lambda p: os.path.abspath(os.path.expanduser(p))
    return {"label": label,
            "base": os.path.join(ex(root), str(label), "autorun"),
            "python": w.get("python") or sys.executable,
            "train_repo": ex(w.get("train_repo") or REPO),
            "eval_repo": ex(w.get("eval_repo") or "~/LDRP_eval")}


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def put_file(src, dst):
    """共有フォルダの添付を置く. 同名で中身の違うファイルがあれば上書きせず止める."""
    if os.path.exists(dst):
        if md5(src) != md5(dst):
            raise RuntimeError("a different file already exists: %s" % dst)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def spawn(cmd, cwd, out_path, env=None):
    """切り離して起動する (worker や launchd が終わっても止まらない)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as out:
        subprocess.Popen(cmd, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         env=env, start_new_session=True)


def start_train(s, job):
    for f in job.get("files") or []:
        put_file(os.path.join(s["base"], f["src"]), os.path.join(s["train_repo"], f["dest"]))
    path = os.path.join(HOME_DIR, "jobs", job["id"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"id": job["id"], "commands": job["commands"],
                   "max_processes": job.get("max_processes") or 1}, f)
    cmd = [s["python"], os.path.join(TOOLS, "autorun_train.py"), path]
    if shutil.which("caffeinate"):          # Mac: 学習の間はスリープしない (-i = アイドルでのスリープを止める)
        cmd = ["caffeinate", "-i"] + cmd
    spawn(cmd, s["train_repo"], os.path.join(HOME_DIR, "logs", job["id"] + ".out"))


def start_eval(s, job):
    er = s["eval_repo"]
    os.makedirs(er, exist_ok=True)
    mark = os.path.join(er, ".autorun_code")
    code = os.path.basename(job["code"])
    have = open(mark).read().strip() if os.path.exists(mark) else ""
    if have != code:
        subprocess.run(["tar", "-xzf", os.path.join(s["base"], job["code"]), "-C", er], check=True)
        with open(mark, "w") as f:
            f.write(code)
    bad = [p for p, h in (job.get("eval_hashes") or {}).items()
           if not os.path.exists(os.path.join(er, p)) or md5(os.path.join(er, p)) != h]
    if bad:
        raise RuntimeError("evaluation code differs after unpacking: %s" % ", ".join(bad))
    for f in job.get("files") or []:
        put_file(os.path.join(s["base"], f["src"]), os.path.join(er, f["dest"]))
    path = os.path.join(HOME_DIR, "jobs", job["id"] + ".json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(job["jobs"], f)
    cmd = [s["python"], "-u", "tools/eval_lare_factor.py", "--jobs-file", path, "-j", str(job.get("n_par") or 1),
           "--tag", job["id"], "--host", s["label"]]
    if shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i"] + cmd
    # 評価は CPU だけで回す (GPU マシンと同じ。結果が Mac と一致することを確認済み)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    spawn(cmd, er, os.path.join(er, "logs", "autorun_%s.out" % job["id"]), env=env)


def take_jobs(s):
    """inbox のジョブを started/ に移してから起動する (移してから起動するので、2 回起動しない)."""
    errors = AR.load_json(ERRORS, [])
    for p in sorted(glob.glob(os.path.join(s["base"], "inbox", "*.json"))):
        job = AR.load_json(p, None)
        if job is None:                     # 同期の途中など。次の回にもう一度読む
            continue
        started = os.path.join(s["base"], "started")
        os.makedirs(started, exist_ok=True)
        dst = os.path.join(started, os.path.basename(p))
        os.replace(p, dst)
        try:
            (start_train if job.get("kind") == "train" else start_eval)(s, job)
            job["started_at"] = AR.now_utc().isoformat()
            log("started %s (%s)" % (job["id"], job.get("kind")))
        except Exception as e:
            job["error"] = str(e)[:500]
            errors = (errors + [{"job": job["id"], "error": job["error"], "at": AR.now_utc().isoformat()}])[-10:]
            log("cannot start %s: %s" % (job["id"], e))
        with open(dst, "w") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
    os.makedirs(HOME_DIR, exist_ok=True)
    with open(ERRORS, "w") as f:
        json.dump(errors, f, ensure_ascii=False)
    return errors


def copy_eval_logs(s):
    src = os.path.join(s["eval_repo"], "logs")
    if os.path.isdir(src):
        subprocess.run(["rsync", "-a", "--include", "*/", "--include", "*.txt", "--exclude", "*",
                        src + "/", os.path.join(s["base"], "eval_logs") + "/"], capture_output=True)


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except (IOError, OSError):
        return None


def write_status(s, errors):
    tr, er = s["train_repo"], s["eval_repo"]
    cfg = os.path.join(tr, "src", "epymarl", "src", "config")
    algs = dict((os.path.basename(p)[:-5], read(p)) for p in glob.glob(os.path.join(cfg, "algs", "*.yaml")))
    lare = []
    for sub in ("models", "checkpoints"):
        lare += [os.path.basename(p) for p in glob.glob(os.path.join(tr, "src", "lare", "path", sub, "*.pth"))]
    head = subprocess.run("git rev-parse --short HEAD 2>/dev/null; git status --porcelain 2>/dev/null | wc -l",
                          shell=True, cwd=tr, capture_output=True, text=True).stdout
    probe = subprocess.run(["bash", "-c", AR.PROBE], capture_output=True, text=True, timeout=60).stdout
    st = {"at": AR.now_utc().isoformat(), "label": s["label"], "python": s["python"],
          "train_repo": tr, "eval_repo": er, "probe": probe, "train_head": " ".join(head.split()),
          "epymarl_config": {"default": read(os.path.join(cfg, "default.yaml")),
                             "gymma": read(os.path.join(cfg, "envs", "gymma.yaml")), "algs": algs},
          "lare_models": sorted(set(lare)), "errors": errors}
    os.makedirs(s["base"], exist_ok=True)
    tmp = os.path.join(s["base"], ".status.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(s["base"], "status.json"))


LABEL = "com.ldrp.autorun-worker"
PLIST = os.path.expanduser("~/Library/LaunchAgents/%s.plist" % LABEL)


def install(interval):
    """launchd に登録する (この Python・このリポジトリのパスで plist を作る).

    AbandonProcessGroup: worker が終わっても、起動した学習・評価を launchd に止めさせない
    """
    out = os.path.join(HOME_DIR, "worker.log")
    os.makedirs(HOME_DIR, exist_ok=True)
    plist = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>%s</string>
  <key>ProgramArguments</key>
  <array><string>%s</string><string>%s</string></array>
  <key>WorkingDirectory</key><string>%s</string>
  <key>StartInterval</key><integer>%d</integer>
  <key>RunAtLoad</key><true/>
  <key>AbandonProcessGroup</key><true/>
  <key>StandardOutPath</key><string>%s</string>
  <key>StandardErrorPath</key><string>%s</string>
</dict>
</plist>
""" % (LABEL, sys.executable, os.path.abspath(__file__), REPO, interval, out, out)
    os.makedirs(os.path.dirname(PLIST), exist_ok=True)
    subprocess.run(["launchctl", "unload", PLIST], capture_output=True)
    with open(PLIST, "w") as f:
        f.write(plist)
    subprocess.run(["launchctl", "load", PLIST], check=True)
    print("installed %s (every %d s). log: %s" % (PLIST, interval, out))


def uninstall():
    subprocess.run(["launchctl", "unload", PLIST], capture_output=True)
    if os.path.exists(PLIST):
        os.remove(PLIST)
    print("uninstalled %s (jobs already started keep running)" % PLIST)


def main(argv=None):
    p = argparse.ArgumentParser(description="Start autorun jobs placed in the shared folder (for machines "
                                            "the commander cannot reach by ssh). Run every few minutes.")
    p.add_argument("--status-only", action="store_true", help="only write status.json (start nothing)")
    p.add_argument("--install", action="store_true", help="register with launchd (macOS) and exit")
    p.add_argument("--interval", type=int, default=300, help="seconds between runs for --install")
    p.add_argument("--uninstall", action="store_true", help="remove the launchd job and exit")
    args = p.parse_args(argv)
    if args.install:
        settings()                          # 設定が無ければここで止める
        return install(args.interval) or 0
    if args.uninstall:
        return uninstall() or 0
    s = settings()
    errors = AR.load_json(ERRORS, []) if args.status_only else take_jobs(s)
    copy_eval_logs(s)
    write_status(s, errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
