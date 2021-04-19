import json
import math, os
import numpy as np


if __name__ == "__main__":
	ray_results_path = "/home/meng/ray_results/PPO"
	#trail_path = "PPO_OpenRoomEnvironmentRLLIB_f8a58_00000_0_2021-04-15_08-40-57"
	trail_path = "PPO_OpenRoomEnvironmentRLLIB_3b6c3_00000_0_2021-04-15_09-18-37"
	json_path = os.path.join(ray_results_path, trail_path, "result.json")
	
	# one line per iteration
	with open(json_path) as f:
		i = 0
		for line in f:
			if i == 0:
				data = json.loads(line) 
				print(data)
			i += 1
		print(i)	