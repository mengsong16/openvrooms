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

from gibson2.utils.utils import quatToXYZW, quatFromXYZW
from transforms3d.euler import euler2quat, quat2euler

class RelocatePointGoalFixedTask(BaseTask):
    """
    Relocate Point Goal Fixed Task
    The goal is to push objects to fixed goal locations
    """

    def __init__(self, env):
        super(RelocatePointGoalFixedTask, self).__init__(env)

        self.reward_type = self.config.get('reward_type', 'l2')

        self.termination_conditions = [
            #MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
            #PointGoal(self.config),
        ]

        self.reward_functions = [
            PotentialReward(self.config),
            #CollisionReward(self.config),
            #PointGoalReward(self.config),
        ]

        # get robot's initial pose
        self.agent_initial_pos = np.array(self.config.get('agent_initial_pos', [0, 0, 0]))
        self.agent_initial_orn = np.array(self.config.get('agent_initial_orn', [0, 0, 0]))  # euler angles: rotatation around x,y,z axis

        # get object intial positions (for scene and visualization)
        self.obj_initial_pos = np.array(self.config.get('obj_initial_pos'))
        self.obj_initial_orn = np.array(self.config.get('obj_initial_orn'))
        self.obj_target_pos = np.array(self.config.get('obj_target_pos'))
        self.obj_target_orn = np.array(self.config.get('obj_target_orn'))


        if self.obj_initial_pos.shape[0] != self.obj_target_pos.shape[0]:
            raise Exception("Initial position list should have the same shape as target position list!")

        self.obj_num = self.config.get('obj_num', 1)
        

        if self.obj_initial_pos.shape[0] != self.obj_num:
            raise Exception("Initial position list should have %d objects, instead of %d !"%(self.obj_num, self.obj_initial_pos.shape[0]))


        print("Number of objects: %d"%(self.obj_num))
        print("Initial x-y positions of objects: \n%s"%self.obj_initial_pos)
        print("Target x-y positions of objects: \n%s"%self.obj_target_pos)

        self.goal_format = self.config.get('goal_format', 'cartesian')

        # distance tolerance for goal reaching
        #self.dist_tol = self.termination_conditions[-1].dist_tol
        self.dist_tol = 0.36

        #self.visual_object_at_initial_target_pos = self.config.get(
        #    'visual_object_at_initial_target_pos', True
        #)
        self.visual_object_visible_to_agent = self.config.get(
            'visual_object_visible_to_agent', False
        )
        
        self.third_person_view = self.config.get("third_person_view", True)

        self.load_visualization(env)

        self.get_loaded_interactive_objects(env)

        # ignore collisions with these interactive objects
        env.collision_ignore_body_b_ids |= set(
            [obj.body_id for obj in self.interactive_objects])
        
        # check validity of initial and target scene
        print("--------------- Check validity of initial and target scene ------------")
        self.check_inital_scene_collision(env)
        self.check_target_scene_collision(env)
        print("--------------------------------------------- ")
        

    def get_loaded_interactive_objects(self, env):
        """
        Get loaded interactive objects from the scene

        :param env: environment instance
        :return: a list of interactive objects
        """
        self.interactive_objects = env.scene.interative_objects

        if len(self.interactive_objects) < 1:
            raise Exception("Interactive objects not loaded in the scene!")
        

    def reset_scene(self, env):
        """
        Task-specific scene reset: reset the interactive objects after scene and agent reset

        :param env: environment instance
        """
        env.scene.reset_interactive_object_poses(self.obj_initial_pos, self.obj_initial_orn)

    def load_visualization(self, env):
        """
        Load visualization, such as initial and target position, shortest path, etc

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        cyl_length = 0.2

        self.initial_pos_vis_objs = []
        for i in list(np.arange(self.obj_num)):
            self.initial_pos_vis_objs.append(VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=[1, 0, 0, 0.3],
                radius=self.dist_tol,
                length=cyl_length,
                initial_offset=[0, 0, cyl_length / 2.0]))

        self.target_pos_vis_objs = []
        for i in list(np.arange(self.obj_num)):
            self.target_pos_vis_objs.append(VisualMarker(
                visual_shape=p.GEOM_CYLINDER,
                rgba_color=[0, 0, 1, 0.3],
                radius=self.dist_tol,
                length=cyl_length,
                initial_offset=[0, 0, cyl_length / 2.0]))

        if self.visual_object_visible_to_agent:
            for i in list(np.arange(self.obj_num)):
                env.simulator.import_object(self.initial_pos_vis_objs[i])
                env.simulator.import_object(self.target_pos_vis_objs[i])
        else:
            for i in list(np.arange(self.obj_num)):
                self.initial_pos_vis_objs[i].load()
                self.target_pos_vis_objs[i].load()

    # compute total distance of objects from initial positions to target positions       
    def get_l2_potential(self, env):
        """
        Get potential based on L2 distance

        :param env: environment instance
        :return: L2 distance to the target position
        """

        total_l2_potential = 0.0

        # x,y distance
        for i, obj in enumerate(self.interactive_objects):
            pos, _ = obj.get_position_orientation()
            total_l2_potential += l2_distance(pos[:2], self.obj_target_pos[i])

        return total_l2_potential

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

    def rot_dist_func():
        return rotation.subtract_euler(goal_state["obj_rot"], current_state["obj_rot"])

    def relative_goal(self, goal_state: dict, current_state: dict) -> dict:
        goal_pos = goal_state["obj_pos"]
        obj_pos = current_state["obj_pos"]

        if self.mujoco_simulation.num_objects == self.mujoco_simulation.num_groups:
            # All the objects are different.
            relative_obj_pos = goal_pos - obj_pos
            relative_obj_rot = self.rot_dist_func(goal_state, current_state)

        else:
            # per object relative pos & rot distance.
            rel_pos_dict = {}
            rel_rot_dict = {}

            def get_rel_rot(target_obj_id, curr_obj_id):
                group_goal_state = {"obj_rot": goal_state["obj_rot"][[target_obj_id]]}
                group_current_state = {
                    "obj_rot": current_state["obj_rot"][[curr_obj_id]]
                }

                if self.args.rot_dist_type == "icp":
                    group_goal_state["icp"] = [goal_state["icp"][target_obj_id]]
                    group_current_state["vertices"] = [
                        current_state["vertices"][curr_obj_id]
                    ]

                return self.rot_dist_func(group_goal_state, group_current_state)[0]

            for group_id, obj_group in enumerate(self.mujoco_simulation.object_groups):
                object_ids = obj_group.object_ids

                # Duplicated objects share the same group id.
                # Within each group we match objects with goals according to position in a greedy
                # fashion. Note that we ignore object rotation during matching.
                if len(object_ids) == 1:
                    object_id = object_ids[0]
                    rel_pos_dict[object_id] = goal_pos[object_id] - obj_pos[object_id]
                    rel_rot_dict[object_id] = get_rel_rot(object_id, object_id)
                else:
                    n = len(object_ids)

                    # find the optimal pair matching through greedy.
                    # TODO: may consider switching to `scipy.optimize.linear_sum_assignment`
                    assert obj_pos.shape == goal_pos.shape
                    dist = np.linalg.norm(
                        np.expand_dims(obj_pos[object_ids], 1)
                        - np.expand_dims(goal_pos[object_ids], 0),
                        axis=-1,
                    )
                    assert dist.shape == (n, n)

                    for _ in range(n):
                        i, j = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
                        rel_pos_dict[object_ids[i]] = (
                            goal_pos[object_ids[j]] - obj_pos[object_ids[i]]
                        )
                        rel_rot_dict[object_ids[i]] = get_rel_rot(
                            object_ids[j], object_ids[i]
                        )
                        # once we select a pair of match (i, j), wipe out their distance info.
                        dist[i, :] = np.inf
                        dist[:, j] = np.inf

            assert (
                len(rel_pos_dict)
                == len(rel_rot_dict)
                == self.mujoco_simulation.num_objects
            )
            rel_pos = np.array(
                [rel_pos_dict[i] for i in range(self.mujoco_simulation.num_objects)]
            )
            rel_rot = np.array(
                [rel_rot_dict[i] for i in range(self.mujoco_simulation.num_objects)]
            )
            assert len(rel_pos.shape) == len(rel_rot.shape) == 2

            # padding zeros for the final output.
            relative_obj_pos = np.zeros(
                (self.mujoco_simulation.max_num_objects, rel_pos.shape[-1])
            )
            relative_obj_rot = np.zeros(
                (self.mujoco_simulation.max_num_objects, rel_rot.shape[-1])
            )
            relative_obj_pos[: rel_pos.shape[0]] = rel_pos
            relative_obj_rot[: rel_rot.shape[0]] = rel_rot

        # normalize angles
        relative_obj_rot = rotation.normalize_angles(relative_obj_rot)
        return {
            "obj_pos": relative_obj_pos.copy(),
            "obj_rot": relative_obj_rot.copy(),
        }

    def goal_distance(self, goal_state: dict, current_state: dict) -> dict:
        relative_goal = self.relative_goal(goal_state, current_state)
        pos_distances = np.linalg.norm(relative_goal["obj_pos"], axis=-1)
        rot_distances = rotation.quat_magnitude(
            rotation.quat_normalize(rotation.euler2quat(relative_goal["obj_rot"]))
        )
        return {
            "relative_goal": relative_goal,
            "obj_pos": np.maximum(
                pos_distances + self.mujoco_simulation.goal_pos_offset, 0
            ),  # L2 dist
            "obj_rot": self.mujoco_simulation.goal_rot_weight
            * rot_distances,  # quat magnitude
        }

    def check_inital_scene_collision(self, env):
        state_id = p.saveState()

        success = env.test_valid_position(env.robots[0],  self.agent_initial_pos,  self.agent_initial_orn)
        if not success:
            print("Initial scene Failed: unable to set robot initial pose without collision.")
        
        for i, obj in enumerate(env.scene.interative_objects):    
            success = env.test_valid_position(obj,  [self.obj_initial_pos[i][0], self.obj_initial_pos[i][1], obj.goal_z],  self.obj_initial_orn[i])
            if not success:
                print("Initial scene Failed: unable to set object %d's initial pose without collision."%(i))
                

        p.restoreState(state_id)
        p.removeState(state_id)
        print("Validity check of initial scene Finished!")

    def check_target_scene_collision(self, env):
        state_id = p.saveState()
        
        for i, obj in enumerate(env.scene.interative_objects):    
            success = env.test_valid_position(obj,  [self.obj_target_pos[i][0], self.obj_target_pos[i][1], obj.goal_z],  self.obj_target_orn[i])
            if not success:
                print("Target scene Failed: unable to set object %d's target pose without collision."%(i))

        p.restoreState(state_id)
        p.removeState(state_id)
        print("Validity check of target scene Finished!")    

    def reset_agent(self, env):
        """
        Task-specific agent reset: land the robot to initial pose, compute initial potential

        :param env: environment instance
        """

        # land robot at initial pose
        env.land(env.robots[0], self.agent_initial_pos, self.agent_initial_orn)
    
        self.robot_pos = self.agent_initial_pos[:2]

        # reset reward functions
        for reward_function in self.reward_functions:
            reward_function.reset(self, env)

    def get_termination(self, env, collision_links=[], action=None, info={}):
        """
        Aggreate termination conditions and fill info
        """
        done, info = super(RelocatePointGoalFixedTask, self).get_termination(
            env, collision_links, action, info)

        return done, info

    # useful when the first person view is adopted
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

        agent = env.robots[0]

        # [x,y,z]
        robot_position = agent.get_position()
        
        # [x,y,z,w] --> [w,x,y,z] --> euler angles
        robot_orientation = np.array(agent.get_orientation())
        robot_orientation = quat2euler(quatFromXYZW(robot_orientation, 'wxyz'))

        # 3d in world frame if third person view is adopted
        robot_linear_velocity = agent.get_linear_velocity()
        
        # 3d in world frame if third person view is adopted
        robot_angular_velocity = agent.get_angular_velocity()
        

        # 12 d in total
        task_obs = np.concatenate((robot_position, robot_orientation, robot_linear_velocity, robot_angular_velocity), axis=None)
        
        
        # object current pose: 6d each
        for obj in self.interactive_objects:
            pos, orn = obj.get_position_orientation()

            orn = np.array(orn)
            orn = quat2euler(quatFromXYZW(orn, 'wxyz'))

            task_obs = np.append(task_obs, pos)
            task_obs = np.append(task_obs, orn)

        
        # object target pose: 6d each
        for i, obj in enumerate(self.interactive_objects):
            target_pos = [self.obj_target_pos[i][0], self.obj_target_pos[i][1], obj.goal_z]
            task_obs = np.append(task_obs, target_pos)

            orn = np.array(self.obj_target_orn[i])
            task_obs = np.append(task_obs, orn)

        print(task_obs.shape)
        return task_obs

    # visualize initial and target positions of the objects
    def step_visualization(self, env):
        """
        Step visualization

        :param env: environment instance
        """
        if env.mode != 'gui':
            return

        for i in list(np.arange(self.obj_num)):
            self.initial_pos_vis_objs[i].set_position([self.obj_initial_pos[i][0], self.obj_initial_pos[i][1], 0])
            self.target_pos_vis_objs[i].set_position([self.obj_target_pos[i][0], self.obj_target_pos[i][1], 0])


    def step(self, env):
        """
        Perform task-specific step: step visualization and aggregate path length

        :param env: environment instance
        """
        self.step_visualization(env)
        new_robot_pos = env.robots[0].get_position()[:2]
        self.robot_pos = new_robot_pos
