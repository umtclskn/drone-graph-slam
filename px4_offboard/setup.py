import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'px4_offboard'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='umut',
    maintainer_email='umut@todo.todo',
    description='PX4 offboard control: arm, takeoff, fly a rectangular loop, and land in SITL.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offboard_control = px4_offboard.offboard_control:main',
            'imu_bridge = px4_offboard.imu_bridge:main',
            'ground_truth_bridge = px4_offboard.ground_truth_bridge:main',
            'ekf2_odometry_adapter = px4_offboard.ekf2_odometry_adapter:main',
        ],
    },
)
