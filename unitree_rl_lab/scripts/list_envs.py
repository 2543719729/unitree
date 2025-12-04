"""
脚本用于打印 Isaac Lab 中所有可用的环境。

该脚本遍历所有已注册的环境并将详细信息存储在表格中。
它会打印环境名称、入口点和配置文件。

所有环境都在 `unitree_rl_lab` 扩展中注册。它们的名称以 `Unitree` 开头。
"""

"""首先启动 Isaac Sim 仿真器。"""


import importlib
import pathlib
import pkgutil
import sys


def _walk_packages(
    path: str | None = None,
    prefix: str = "",
    onerror=None,
):
    """递归地生成路径上的所有模块信息，如果路径为 None，则生成所有可访问的模块信息。

    注意：
        此函数是原始 ``pkgutil.walk_packages`` 函数的修改版本。
        更多详细信息请参考原始 ``pkgutil.walk_packages`` 函数。
    """

    def seen(p, m={}):
        if p in m:
            return True
        m[p] = True  # noqa: R503

    # 遍历指定路径下的模块
    for info in pkgutil.iter_modules(path, prefix):

        # 生成模块信息
        yield info

        # 如果是包，则递归遍历
        if info.ispkg:
            try:
                __import__(info.name)
            except Exception:
                if onerror is not None:
                    onerror(info.name)
                else:
                    raise
            else:
                path = getattr(sys.modules[info.name], "__path__", None) or []

                # 不要遍历之前已经见过的路径项
                path = [p for p in path if not seen(p)]

                # 递归调用
                yield from _walk_packages(path, info.name + ".", onerror)


def import_packages():
    """导入必要的包以注册环境。"""
    # 将任务目录添加到系统路径中，以便可以导入其中的模块
    sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}/source/unitree_rl_lab/unitree_rl_lab/tasks/")
    # 遍历并导入 locomotion 和 mimic 机器人相关的包
    for package in ["locomotion.robots", "mimic.robots"]:
        package = importlib.import_module(package)
        # 递归遍历包中的所有模块并导入，这会触发环境的注册
        for _ in _walk_packages(package.__path__, package.__name__ + "."):
            pass
    # 从系统路径中移除添加的目录
    sys.path.pop(0)


# 执行导入操作，确保所有环境都被注册
import_packages()

"""其余部分紧随其后。"""

import gymnasium as gym
from prettytable import PrettyTable


def main():
    """打印所有在 `unitree_rl_lab` 扩展中注册的环境。"""
    # 创建一个表格用于显示环境信息
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Unitree RL Lab" # 表格标题：Unitree RL Lab 中可用的环境
    # 设置表格列的对齐方式
    table.align["Task Name"] = "l"   # 任务名称左对齐
    table.align["Entry Point"] = "l" # 入口点左对齐
    table.align["Config"] = "l"      # 配置文件左对齐

    # 环境计数器
    index = 0
    # 获取所有 Isaac 环境名称
    for task_spec in gym.registry.values():
        # 筛选出包含 "Unitree" 且不包含 "Isaac" 的任务（即 Unitree 扩展提供的任务）
        if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
            # 将环境详细信息添加到表格中
            # 包含：序号、任务ID、入口点函数、配置文件入口点
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            # 增加计数
            index += 1

    # 打印表格
    print(table)


if __name__ == "__main__":
    try:
        # 运行主函数
        main()
    except Exception as e:
        raise e
