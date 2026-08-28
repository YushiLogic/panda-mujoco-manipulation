import sys

import mujoco
import numpy as np
import scipy

import panda_mujoco


MODEL_XML = """
<mujoco model="environment_check">
    <option timestep="0.002" gravity="0 0 -9.81"/>

    <worldbody>
        <geom type="plane" size="1 1 0.1"/>

        <body name="ball" pos="0 0 0.5">
            <freejoint/>
            <geom type="sphere" size="0.05" mass="1.0"/>
        </body>
    </worldbody>
</mujoco>
"""


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("MuJoCo:", mujoco.__version__)
    print("NumPy:", np.__version__)
    print("SciPy:", scipy.__version__)
    print("Project package:", panda_mujoco.__file__)

    model = mujoco.MjModel.from_xml_string(MODEL_XML)
    data = mujoco.MjData(model)

    for _ in range(1000):
        mujoco.mj_step(model, data)

    if not np.all(np.isfinite(data.qpos)):
        raise RuntimeError("Simulation produced invalid joint states.")

    print("Simulation time:", round(data.time, 3), "s")
    print("Ball height:", round(float(data.qpos[2]), 4), "m")
    print("Environment check: PASSED")


if __name__ == "__main__":
    main()