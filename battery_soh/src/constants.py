"""枚举常量：AnchorType / Tier / Flag / RMode。

铁律（方案第二章）：
  禁止 (剩余容量_end - 剩余容量_start) / (SOC_end - SOC_start) —— 恒等式回声。
  允许 剩余容量 / SOC_anchor，且仅当该点被电压或业务事实锚定。
"""
from __future__ import annotations

from enum import Enum


class AnchorType(str, Enum):
    """四个锚点（方案 3.3）"""

    FULL_END = "FULL_END"      # A. 满充末端：SOC>0.95 且 V>V_full*0.99 且 0<I<死区 持续>=2h
    SWAP_FULL = "SWAP_FULL"    # B. 站内满充完成（换电场景独有，最干净）：I 正->静置 + SOC>=0.98
    EMPTY_END = "EMPTY_END"    # C. 放空末端：SOC<0.10 且 V<V_cutoff*1.05 且 I<0
    OCV = "OCV"                # D. 静置 OCV：|I|<0.02C 持续>=2h，末段电压查 OCV 表
    NONE = "NONE"              # 其余所有点：不取值（Tier-3，禁入统计）


class Tier(int, Enum):
    """置信度分级（方案第四章 / 5-4c）"""

    TIER1 = 1   # 评分>=30 且无严重异常且 7 天内有另一锚点支撑 → 进 SOH 统计/退役判定
    TIER2 = 2   # 评分 10~29 或单侧支撑 → 保留观察，权重 0.3
    TIER3 = 3   # 评分<10 或有严重异常 → 严禁入统计，仅绘图与回溯


class Flag(str, Enum):
    """异常标记（方案 5-4a，超界不删、只打标记）"""

    OUT_OF_RANGE = "out_of_range"                  # 容量越界
    VOLTAGE_INCONSISTENT = "voltage_inconsistent"  # 电压不自洽
    TEMP_OUT = "temp_out"                          # 温度异常
    ANCHOR_CONFLICT = "anchor_conflict"            # 锚点冲突
    SENSOR_DRIFT = "sensor_drift"                  # 电流传感器漂移
    CONTRADICTION = "contradiction"                # 前后矛盾（与相邻锚点差异>30%）
    INVALID_LOOPNUM = "invalid_loopnum"            # 循环次数非数值
    INVALID_SENSOR = "invalid_sensor"              # 电压/温度/电流非数值
    TIME_GAP = "time_gap"                          # 采样断档 >6h
    CURRENT_REVERSED = "current_reversed"          # 电流符号与协议相反（已取反修正）


class RMode(str, Enum):
    """Step 0 输出：R = 剩余容量/SOC 的模式（方案 5-Step0.7 / 9.3）"""

    DYNAMIC = "DYNAMIC"    # BMS 在做容量重估 → 正常走 Step 1~8
    CONSTANT = "CONSTANT"  # 纯共线 → 检查锚点密度，不足则降级 L2


class Route(str, Enum):
    """运行路线（Step 0 输出）"""

    ABSOLUTE = "ABSOLUTE"  # 绝对容量法（Step 1~8）
    L2_TREND = "L2_TREND"  # 降级：电压曲线形态漂移 / 群体排序，不给 SOH 数值
    L3_RANK = "L3_RANK"    # 降级：仅剩余容量趋势方向，仅供排序


# 中文原始字段名（附录 A）
COL_TABLE = "表名"
COL_DEVICE = "电池id"
COL_CHANNEL = "渠道号"
COL_LOOP = "循环次数"
COL_CAP = "剩余容量"
COL_SOC = "SOC"
COL_VOLT = "电压"
COL_CURR = "电流"
COL_TEMP = "温度"
COL_TIME = "更新时间"

RAW_COLUMNS = [
    COL_TABLE, COL_DEVICE, COL_CHANNEL, COL_LOOP, COL_CAP,
    COL_SOC, COL_VOLT, COL_CURR, COL_TEMP, COL_TIME,
]
