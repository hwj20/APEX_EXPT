import mujoco
import mujoco.viewer as viewer
import numpy as np
import os
import cv2

def get_camera_id(model, camera_name):
    """返回名字为 camera_name 的摄像机 ID，否则返回 0（默认）"""
    # camera_name 在内部是 bytes，所以要 encode 一下
    name_bytes = camera_name.encode()
    for cam_id in range(model.ncam):
        # 利用 mj_id2name 获取每个摄像机的名字
        cam_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        if cam_name == name_bytes:
            return cam_id
    print(f"Camera '{camera_name}' not found. Using default camera (0).")
    return 0

def save_screenshot(scene_xml, save_path, camera_name='top', width=640, height=480):
    model = mujoco.MjModel.from_xml_string(scene_xml)
    data  = mujoco.MjData(model)

    # 把场景“激活”一下
    mujoco.mj_forward(model, data)
    # 或者你想让杯子真的开始下落就：
    # for _ in range(10):
    #     mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, width=width, height=height)

    renderer.update_scene(data, camera=camera_name)
    pixels = renderer.render()
    screenshot = np.flipud(pixels)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))



def list_contents(scene_xml):
    model = mujoco.MjModel.from_xml_string(scene_xml)
    # body names
    bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
              for i in range(model.nbody)]
    # geom names (没有显式 name 时，MuJoCo 会用 ""，所以我们也列一下类型和所属 body 以防万一)
    geoms = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        body = bodies[model.geom_bodyid[i]]
        geoms.append(f"{name or '<no-name>'} (on body '{body}')")
    print("Bodies:", bodies)
    print("Geoms: ", geoms)

if __name__ == '__main__':
    # 示例：加载三个场景并保存截图
    from scene import scene1, scene2, scene3

    list_contents(scene1)

    save_screenshot(scene1, "screenshots/cup_falling.png", camera_name='top')
    save_screenshot(scene2, "screenshots/cat_approach.png", camera_name='top')
    save_screenshot(scene3, "screenshots/agent_path_selection.png", camera_name='top')