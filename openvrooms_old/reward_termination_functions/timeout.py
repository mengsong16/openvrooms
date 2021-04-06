from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction


class Timeout(BaseRewardTerminationFunction):
    """
    Timeout
    Episode terminates if max_step steps have passed
    """

    def __init__(self, config):
        super(Timeout, self).__init__(config)
        self.max_step = self.config.get('max_step', 500)
        self.reward = float(self.config.get('time_elapse_reward', -0.01))

    def get_reward_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = env.current_step >= self.max_step
        success = False

        return self.reward, done, success

    def get_name(self):
        return "timeout"    