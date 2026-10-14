"""Export exact compiled ELF3 body/geom frames without meshes or dynamics."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mujoco
from umr import runtime


def main():
    root = Path(__file__).resolve().parents[1]
    package = root.parents[1]
    xml = package / "robot/elf3_lite/xml/elf3.xml"
    model = runtime.load_robot(xml, use_fast=False)
    def floats(values):
        return " ".join(format(float(x), ".17g") for x in values)
    result = ET.Element("mujoco", model="elf3_umr_kinematics")
    ET.SubElement(result, "compiler", angle="radian", autolimits="true")
    world = ET.SubElement(result, "worldbody")
    bodies = {0: world}
    for index in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        body = ET.SubElement(bodies[int(model.body_parentid[index])], "body", name=name,
                             pos=floats(model.body_pos[index]), quat=floats(model.body_quat[index]))
        bodies[index] = body
        if model.body_mass[index] > 0:
            ET.SubElement(body, "inertial", mass=format(float(model.body_mass[index]), ".17g"),
                          pos=floats(model.body_ipos[index]), quat=floats(model.body_iquat[index]),
                          diaginertia=floats(model.body_inertia[index]))
        for joint in range(int(model.body_jntadr[index]), int(model.body_jntadr[index] + model.body_jntnum[index])):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            kind = int(model.jnt_type[joint])
            if kind == int(mujoco.mjtJoint.mjJNT_FREE):
                ET.SubElement(body, "freejoint", name=name)
            else:
                attrs = dict(name=name, type="hinge" if kind == int(mujoco.mjtJoint.mjJNT_HINGE) else "slide",
                             pos=floats(model.jnt_pos[joint]), axis=floats(model.jnt_axis[joint]),
                             ref=format(float(model.qpos0[int(model.jnt_qposadr[joint])]), ".17g"),
                             limited="true" if model.jnt_limited[joint] else "false")
                if model.jnt_limited[joint]:
                    attrs["range"] = floats(model.jnt_range[joint])
                ET.SubElement(body, "joint", **attrs)
        for geom in range(int(model.body_geomadr[index]), int(model.body_geomadr[index] + model.body_geomnum[index])):
            ET.SubElement(body, "geom", type="sphere", size="0.001", density="0",
                          contype="0", conaffinity="0", pos=floats(model.geom_pos[geom]),
                          quat=floats(model.geom_quat[geom]))
    ET.indent(result)
    ET.ElementTree(result).write(root / "elf3_kinematics.xml", encoding="unicode")
    (root / "elf3_kinematics.sha256").write_text(runtime.robot_signature(xml) + "\n")
    print(f"Exported {model.nbody} body frames and {model.njnt} joints")


if __name__ == "__main__":
    main()
