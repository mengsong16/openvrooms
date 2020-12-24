from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.termination_conditions.point_goal import PointGoal
from gibson2.reward_functions.potential_reward import PotentialReward
from gibson2.reward_functions.collision_reward import CollisionReward
from gibson2.reward_functions.point_goal_reward import PointGoalReward

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject

from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.objects.interactive_object import InteractiveObj

import numpy as np


class RelocatePointGoalFixedTask(BaseTask):
    """
    Relocate Point Goal Fixed Task
    The goal is to push objects to fixed goal locations
    """

    def __init__(self, env):
        super(PointNavFixedTask, self).__init__(env)

        self.reward_type = self.config.get('reward_type', 'l2')

        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            PointGoal(self.config),
        ]

        self.reward_functions = [
            PotentialReward(self.config),
            CollisionReward(self.config),
            PointGoalReward(self.config),
        ]

        self.initial_pos = np.array(self.config.get('initial_pos', [0, 0, 0]))
        self.initial_orn = np.array(self.config.get('initial_orn', [0, 0, 0]))

        self.target_pos = np.array(self.config.get('target_pos', [5, 5, 0]))
        self.goal_format = self.config.get('goal_format', 'polar')
        self.dist_tol = self.termination_conditions[-1].dist_tol

        self.visual_object_at_initial_target_pos = self.config.get(
            'visual_object_at_initial_target_pos', True
        )
        self.target_visual_object_visible_to_agent = self.config.get(
            'target_visual_object_visible_to_agent', False
        )
        self.floor_num = 0

        self.load_visualization(env)

        self.interactive_objects = self.load_interactive_objects(env)
        env.collision_ignore_body_b_ids |= set(
            [obj.body_id for obj in self.interactive_objects])
        

    def load_interactive_objects(self, env):
        """
        Load interactive objects (YCB objects)

        :param env: environment instance
        :return: a list of interactive objects
        """
        interactive_objects = []
        
        return interactive_objects

    def reset_interactive_objects(self, env):
        """
        Reset the poses of interactive objects to have no collisions with the scene or the robot

        :param env: environment instance
        """
        max_trials = 100

        for obj in self.interactive_objects:
            # TODO: p.saveState takes a few seconds, need to speed up
            state_id = p.saveState()
            for _ in range(max_trials):
                _, pos = env.scene.get_random_point(floor=self.floor_num)
                orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
                reset_success = env.test_valid_position(obj, pos, orn)
                p.restoreState(state_id)
                if reset_success:
                    break

            if not reset_success:
                print("WARNING: Failed to reset interactive obj without collision")

            env.land(obj, pos, orn)

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the interactive objects after scene and agent reset

        :param env: environment instance
        """
        super(InteractiveNavRandomTask, self).reset_scene(env)
        self.reset_interactive_objects(env)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        cyl_length = 0.2
        self.initial_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[1, 0, 0, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])
        self.target_pos_vis_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0])

        if self.target_visual_object_visible_to_agent:
            env.simulator.import_object(self.initial_pos_vis_obj)
            env.simulator.import_object(self.target_pos_vis_obj)
        else:
            self.initial_pos_vis_obj.load()
            self.target_pos_vis_obj.load()

        

    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """
        return l2_distance(env.robots[0].get_position()[:2],
                           self.target_pos[:2])

    def get_potential(self, env):
        """
        Compute task-specific potential: distance to the goal

        :param env: environment instance
        :return: task potential
        """
        if self.reward_type == 'l2':
            return self.get_l2_potential(env)
        elif self.reward_type == 'geodesic':
            print("Not implemented!")
            return

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset scene objects or floor plane

        :param env: environment instance
        """
        if isinstance(env.scene, InteractiveIndoorScene):
            env.scene.reset_scene_objects()
        elif isinstance(env.scene, StaticIndoorScene):
            env.scene.reset_floor(floor=self.floor_num)

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """

        # land robot at initial pose
        env.land(env.robots[0], self.initial_pos, self.initial_orn)
    
        self.robot_pos = self.initial_pos[:2]

        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(PointNavFixedTask, self).get_termination(
            env, collision_links, action, info)

        return done, info

    def global_to_local(self, env, pos):
        """
        Convert a 3D point in global frame to agent's local frame

        :param env: environment instance
        :param pos: a 3D point in global frame
        :return: the same 3D point in agent's local frame
        """
        return rotate_vector_3d(pos - env.robots[0].get_position(),
                                *env.robots[0].get_rpy())

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        task_obs = self.global_to_local(env, self.target_pos)[:2]
        if self.goal_format == 'polar':
            task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # linear velocity along the x-axis
        linear_velocity = rotate_vector_3d(
            env.robots[0].get_linear_velocity(),
            *env.robots[0].get_rpy())[0]
        
        # angular velocity along the z-axis
        angular_velocity = rotate_vector_3d(
            env.robots[0].get_angular_velocity(),
            *env.robots[0].get_rpy())[2]
        
        task_obs = np.append(
            task_obs, [linear_velocity, angular_velocity])

        return task_obs

    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        self.initial_pos_vis_obj.set_position(self.initial_pos)
        self.target_pos_vis_obj.set_position(self.target_pos)


    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.path_length += l2_distance(self.robot_pos, new_robot_pos)
        self.robot_pos = new_robot_pos
