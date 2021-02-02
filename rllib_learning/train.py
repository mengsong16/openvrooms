def test_rllib_env():
    # training
    config = {
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
            "env": "navigate",
            "config_file": 'turtlebot_navigate.yaml',
            "mode": "headless",
            "device_idx": 0
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 2,
        "lr": 1e-4, # try different lrs
        "num_workers": 1,  # parallelism
        "framework": "torch"
    }
    '''
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    '''
    ray.init()
    
    trainer = ppo.PPOTrainer(env=OpenRoomEnvironmentRLLIB, config=config)
    

    while True:
        print(trainer.train())

    #results = tune.run(args.run, config=config, stop=stop)

if __name__ == "__main__":  
    #test_all_env()   
    test_rllib_env()  