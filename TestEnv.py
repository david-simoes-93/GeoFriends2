from simulator.GymEnvGF import GymEnvGF

environment = GymEnvGF(rectangle=False, circle=True, frameskip=10, air_movement=False, square_interrupt_growth=False)

for trial_number in range(3):
    observation = environment.reset()
    environment.render()

    while True:
        action = environment.action_space.sample()                                          # take a random action
        observation, reward, terminal, additional_information = environment.step(action)    # step
        environment.render()

        if terminal:
            break

environment.close()
