"""心跳告警契约:成功才刷新、超龄才告警、无记录视同超龄。

锁定 heartbeat 层行为;cmd_index 的"扫描 0 份视同失败不刷心跳"在 cli 层,
由人工验收覆盖(单测里跑 cmd_index 会扫到真实 ~/.cmem/raw,不适合)。
"""

import time

from cmem.heartbeat import STALE_AFTER_H, describe, mark_success, warn_if_stale


def test_fresh_heartbeat_stays_silent(tmp_path, capsys):
    hb = tmp_path / "last-index-ok"
    mark_success(hb)
    assert warn_if_stale(hb) is False
    assert capsys.readouterr().err == ""
    assert "分钟前" in describe(hb)


def test_stale_and_missing_both_warn(tmp_path, capsys):
    hb = tmp_path / "last-index-ok"
    assert warn_if_stale(hb) is True  # 无记录 = 不知道停摆多久,保守告警
    mark_success(hb)
    later = time.time() + (STALE_AFTER_H + 24) * 3600
    assert warn_if_stale(hb, now=later) is True
    err = capsys.readouterr().err
    assert err.count("⚠️") == 2 and "launchd" in err
    assert "天前" in describe(hb, now=later)
