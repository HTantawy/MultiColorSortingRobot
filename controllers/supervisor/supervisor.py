from controller import Supervisor

supervisor = Supervisor()

vacuum_device_node = supervisor.getFromDef('vacuum')

while supervisor.step(8) != -1:
    # check if the vacuum device node exists.
    if vacuum_device_node:
        # ---IMPORTANT---
        # set the location of the vacuum device to zero
        # to counteract the physics force pushing the gripper away
        # Physics node is unfortunatelly needed for the vacuum device
        # ---IMPORTANT---
        vacuum_device_node.getField('translation').setSFVec3f([0, 0, 0.02])
    else:
        print("Vacuum device not found.")