from simulator.GymEnvGF import GymEnvGF

env = GymEnvGF(rectangle=True, circle=False)

for trial_number in range(3):
    obs, info = env.reset()
    print(info)
    print(env.action_space.spaces)

    while True:
        #print(obs[1])

        env.render()
        obs, reward, term, info = env.step(env.action_space.sample())     # take a random action

        if term:
            break

env.close()
