import os
import sys

# conda 环境的 C++ 依赖 DLL（zlib/lodepng 等）存放在 <环境根>/Library/bin，
# 只在 `conda activate` 激活后才会进入 PATH。直接用 python.exe 全路径运行时
# Windows 会按 PATH 搜到别处软件的老版本同名 DLL，导致渲染时原生崩溃（exit 127）。
# 这里从解释器位置自动推导并前置，等价于激活环境。
os.environ['PATH'] = os.path.join(os.path.dirname(sys.executable), 'Library', 'bin') + ';' + os.environ['PATH']

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("Agg")   # 纯软件渲染：只写文件不弹窗，避开与 viewer 的 GUI 冲突
import matplotlib.pyplot as plt
from panda_mujoco.simulation import PandaScene, PANDA_HOME_QPOS
from panda_mujoco.joint_trajectory import JointPath

def main():
    """演示 MoveJ（关节空间移动）规划器的使用。

    机械臂从 home → 左 → 右 → home，途中记录每个控制拍的末端位置误差，
    最后画出误差曲线。
    """
    CONTROL_EVERY=10 #物理步：控制拍 = 500Hz：50Hz 
    MOVE_TICKS   = 150   # 每段运动持续的控制拍（150拍 ≈ 3秒）
    SETTLE_TICKS = 150   # 全部走完后保持的控制拍数

    HOME=PANDA_HOME_QPOS[:7] #home 关节角（7维）
    LEFT=[0.6,0.0,0.0,-1.57079,0.0,3.0,-1.7853] #底座左转
    RIGHT=[-0.6,0.0,0.0,-1.57079,0.0,3.0,-1.7853] #底座右转

    scene = PandaScene() #构造时已自动 reset 到 home
    plan=JointPath([HOME,LEFT,RIGHT,HOME],ticks_per_move=MOVE_TICKS) #规划器：home→左→右→home
    err_log=[] #误差曲线：每个控制拍的末端位置误差
    last_seg,settle,done,step=0,0,False,0

    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        while viewer.is_running() and not done:
            if step % CONTROL_EVERY == 0: #每 CONTROL_EVERY 个物理步推进一次控制拍
                target=plan.target() #取当前插值目标
                scene.data.ctrl[:7]=target #取当前插值目标发给电机
                err_log.append([scene.data.time, *(target - scene.data.qpos[:7])])  # ✓ 注意最外层的 [ ] #记录误差曲线
                plan.advance() #推进一个控制拍
                if plan.seg !=last_seg: #段切换时重置 settle 计数器
                    goal=plan.goals[last_seg+1] #上一段的目标
                    err=np.max(np.abs(goal-scene.data.qpos[:7])) #上一段的最大误差
                    name='LEFT'if last_seg==0 else 'RIGHT' if last_seg==1 else 'HOME'
                    print(f"第 {last_seg}段完成：{name}，最大误差={err:.4f}rad，t={scene.data.time:.1f}s")
                    last_seg=plan.seg
                if plan.finished(): #所有段都走完了，开始 settle
                    settle+=1
                    if settle>=SETTLE_TICKS: #settle 够久了，结束 demo
                        done=True                   #done 后窗口自动关闭
            mujoco.mj_step(scene.model, scene.data) #推进物理
            viewer.sync()
            step+=1

    # ---- 主循环结束后：保存关节误差曲线（在 main 内部，才能访问 err_log）----
    os.makedirs('outputs', exist_ok=True)
    arr = np.array(err_log)                      # N×8 表格：第0列时间，第1~7列各关节误差
    plt.figure(figsize=(8, 4))
    for j in range(7):
        plt.plot(arr[:, 0], np.abs(arr[:, 1 + j]), label=f'joint{j+1}')
    plt.xlabel('time (s)'); plt.ylabel('|error| (rad)')
    plt.title('joint tracking error (Day4)')
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig('outputs/day4_joint_error_curve.png', dpi=150)
    print('误差曲线已保存: outputs/day4_joint_error_curve.png')

if __name__ == "__main__":
    main()