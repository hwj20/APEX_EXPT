# MuJoCo Scene Descriptions for APEX

# Scene 1: Cup Falling from Table
scene1 = '''
<mujoco>
    <worldbody>
        <camera name="top" pos="0 0 5" xyaxes="1 0 0 0 -1 0"/>
        <body name='table' pos='0 0 0'>
            <geom size='0.5 0.5 0.05' type='box' rgba='0.6 0.3 0.2 1'/>
        </body>
        <body name='cup' pos='0.6 0 0.6'>
            <geom size='0.1 0.1 0.15' type='cylinder' rgba='1 1 1 1'/>
            <joint name='fall' type='slide' axis='0 0 -1'/>
        </body>
    </worldbody>
</mujoco>
'''

# Scene 2: Cat Approaching Agent
scene2 = '''
<mujoco>
    <worldbody>
        <camera name="top" pos="0 0 5" xyaxes="1 0 0 0 -1 0"/>
        <body name='agent' pos='1 0 0'>
            <geom size='0.2 0.2 0.2' type='sphere' rgba='0 0 1 1'/>
        </body>
        <body name='cat' pos='0 0 0'>
            <geom size='0.3 0.2 0.2' type='sphere' rgba='0.5 0.5 0.5 1'/>
            <joint name='approach' type='slide' axis='1 0 0'/>
        </body>
    </worldbody>
</mujoco>
'''

# Scene 3: Agent Path Selection
scene3 = '''
<mujoco>
    <worldbody>
        <camera name="top" pos="0 0 5" xyaxes="1 0 0 0 -1 0"/>
        <body name='table' pos='0 0 0'>
            <geom size='0.5 0.5 0.05' type='box' rgba='0.6 0.3 0.2 1'/>
        </body>
        <body name='child' pos='0 1 0'>
            <geom size='0.3 0.3 0.3' type='sphere' rgba='1 0.5 0.2 1'/>
        </body>
        <body name='exit' pos='0 -1 0'>
            <geom size='0.2 0.2 0.2' type='box' rgba='0 1 0 1'/>
        </body>
        <body name='agent' pos='0 0 0'>
            <geom size='0.2 0.2 0.2' type='sphere' rgba='0 0 1 1'/>
            <joint name='move' type='slide' axis='0 1 0'/>
        </body>
    </worldbody>
</mujoco>
'''
