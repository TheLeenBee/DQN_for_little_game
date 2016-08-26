"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague
 RUN THIS: ./run_nips.py --rom Breakout
"""
import logging
import numpy as np
import cv2
import mini_juego as mj
import matplotlib.pyplot as plt
import pickle

# Number of rows to crop off the bottom of the (downsampled) screen.
# This is appropriate for breakout, but it may need to be modified
# for other games.
CROP_OFFSET = 8


class ALEExperiment(object):
    def __init__(self, ale, agent, resized_width, resized_height,
                 resize_method, num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng):
        self.ale = ale
        print('ale', ale)
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode

        # #################
        # Change: put in our set of actions (0 to 3)
        # self.min_action_set = ale.getMinimalActionSet()
        # minaction set array([0, 1, 3, 4], dtype=int32)
        self.min_action_set = np.array((0, 1, 2, 3))
        # #################

        self.resized_width = resized_width
        self.resized_height = resized_height
        self.resize_method = resize_method
        self.width, self.height = ale.getScreenDims()  # ##
        # We probs don't care because we have our own images that don't need resizing

        self.buffer_length = 2
        self.buffer_count = 0

        # #################
        # Change: set the height to be 84, might not be necessary
        # self.screen_buffer = np.empty((self.buffer_length,
        #                               self.height, self.width),
        #                              dtype=np.uint8)

        self.screen_buffer = np.empty((self.buffer_length,
                                       84, 84),
                                      dtype=np.uint8)
        # #################

        self.total_steps = 0
        self.time_step = 0
        self.mean_reward = []
        self.steps_per_episode = 100
        self.collected_rewards = 0
        self.position_collection = {}
        self.random_init = 0
        self.two_types = 0
        self.mod = 10  # We test every mod epochs

        # #################
        # Parameters:
        self.num_epochs = 1001  # Each epoch starts a new game of 100 time steps for training and 200 for testing
        self.number_of_tests = 10  # How many games do we run at every testing step
        self.repetitions_experiment = 20  # Repetitions of the same environment with separately retrained nets
        self.name_file = 'Richtig_Repeat_indi_longestTest'

        # Other (Initialise stuff)
        self.runs = self.num_epochs / self.mod
        self.good_vs_bad_all = []
        self.good_vs_bad = np.zeros((1, 2))
        self.ep = 0
        self.testing = 0
        self.change_dir = 0
        # #################

        self.terminal_lol = False  # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        #  @Kai: what I changed the most in this function is basically the repetitions of the different experiment and
        # saving the results of all the runs at the end.

        overallC = 0
        for random in range(1,2):
            self.random_init = random
            for twotypes in range(1):
                counter_experiment = 0
                good_vs_bad_over_reps = []
                all_scores_over_reps = []
                for reps in range(self.repetitions_experiment):
                    self.two_types = twotypes
                    for i in range(self.number_of_tests):
                        if not self.random_init:
                            # Change here to reload
                            self.position_collection[i] = mj.initialise_controlled_items(self.two_types)
                        else:
                            self.position_collection[i] = mj.initialise_items(self.two_types)

                    overallC += 1
                    results = []
                    x = []
                    counter = 0

                    for epoch in range(1, self.num_epochs + 1):
                        # #################
                        # Parameters:
                        self.epoch_length = 100
                        self.steps_per_episode = 100
                        # #################

                        if epoch % self.mod == 1:  # Test every mod epochs
                            # Initialise for testing
                            self.testing = 1
                            self.agent.episode_reward = 0
                            self.collected_rewards = np.zeros((self.number_of_tests, 1))
                            self.agent.total_reward = 0
                            self.agent.start_testing()
                            self.good_vs_bad = np.zeros((1, 2))
                            # Test several times:
                            for test_epoch in range(self.number_of_tests):
                                self.epoch_length = 200
                                self.steps_per_episode = 200
                                print 'Running test epoch ', test_epoch
                                self.run_epoch(test_epoch, self.epoch_length, True)
                                self.collected_rewards[test_epoch] = self.agent.episode_reward
                            # Save all the test results
                            self.good_vs_bad_all.append(self.good_vs_bad)
                            res = np.array((0, 0))
                            res[0] = np.mean(self.collected_rewards)
                            res[1] = np.std(self.collected_rewards)
                            results.append(self.collected_rewards)
                            x.append(epoch)  # Also save the time steps at which we test for the x-axis in the plot
                            counter += 1
                            self.agent.finish_testing(epoch)
                            self.testing = 0

                        # Training
                        print('Epoch ' + str(epoch))
                        self.run_epoch(epoch, self.epoch_length)
                        self.agent.finish_epoch(epoch)
                        self.ep = epoch

                    # Save results in a pickle file
                    with open('resultsNeu/'+str(self.name_file)+'_London_testTS_' + str(self.number_of_tests) + '_testLength_' + str(self.epoch_length) +
                              '2types-' + str(self.two_types) + '_random-' + str(self.random_init) + '_' + str(reps)+ '.pickle', 'w') as f:
                        pickle.dump([self.good_vs_bad_all, results, x], f)

                    # Save the results over the different trained DQNs
                    all_scores_over_reps_temp = np.zeros((len(results), self.number_of_tests))
                    good_vs_bad_over_reps_temp = np.zeros((len(results), 2))
                    for time_step in range(len(results)):
                        all_scores_over_reps_temp[time_step, :] = results[time_step].T
                        good_vs_bad_over_reps_temp[time_step, :] = self.good_vs_bad_all[time_step]
                    if counter_experiment == 0:
                        all_scores_over_reps = all_scores_over_reps_temp
                        good_vs_bad_over_reps = good_vs_bad_over_reps_temp
                        all_scores_over_reps = np.column_stack((all_scores_over_reps, all_scores_over_reps_temp))
                        good_vs_bad_over_reps += good_vs_bad_over_reps_temp
                    counter_experiment += 1

                with open('resultsNeu/All_' + str(self.name_file) + '_twoT_' + str(self.two_types) + '_rand_' +
                          str(self.random_init) + '_reps-' + str(self.number_of_tests) + '.pickle',
                          'w') as f:
                    pickle.dump([good_vs_bad_over_reps, all_scores_over_reps, x], f)

                f.close()


    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False  # Make sure each epoch starts with a reset.
        steps_left = num_steps

        # Run epoch for several steps
        counter = 0
        while steps_left > 0:
            self.terminal_lol = False
            self.total_steps += self.time_step
            self.time_step = 0
            counter += 1
            _, num_steps = self.run_episode(steps_left, testing, epoch)

            # Marta
            steps_left -= self.time_step

    def _init_episode(self, testing, epoch):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        # #############
        # Some changes to this one as well, to restar in our game environment
        if not self.terminal_lol or self.ale.game_over():
            self.ale.reset_game()
            if testing:
                self.positions = self.position_collection[epoch]
            else:
                if self.random_init:
                    self.positions = mj.initialise_items(self.two_types)  # Initialise our game
                else:
                    self.positions = mj.initialise_controlled_items(self.two_types)  # Initialise game with grid
            self.time_step = 0

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self._act(0, testing)  # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self._act(0, testing)
        self._act(0, testing)

    def _act(self, action, testing):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        # #######################################
        # Changed this to return the reward of our game
        # index = self.buffer_count % self.buffer_length
        # reward = self.ale.act(action)
        reward, self.positions, self.time_step, self.change_dir = mj.get_consecutive_frames(action, self.positions, self.time_step)
        # Count whether the agent picked up a positive or a negative reward.
        if reward < 0:
            # Negative reward, bad!
            self.good_vs_bad[0, 0] += 1
        elif reward > 0:
            # POsiive reward, good!
            self.good_vs_bad[0, 1] += 1

        screen = mj.get_screen(self.positions)

        # You can viualise here:
        #myobj, self.ax, self.time_step = mj.plot_screen(screen, self.ax, self.time_step)

        index = self.buffer_count % self.buffer_length

        # This gets the greyscale image but what for? It does not assign it to anything
        # Is it saving it in screen_buffer. The buffer just saves two frames a before and an after one
        # once the action is taken
        np.set_printoptions(threshold=np.nan)
        # self.ale.getScreenGrayscale(self.screen_buffer[index, ...])
        self.screen_buffer[index, ...] = screen
        # print(self.screen_buffer[index, ...])
        # raw_input()

        self.buffer_count += 1
        return reward

    def _step(self, action, testing):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self._act(action, testing)

        return reward

    def run_episode(self, max_steps, testing, epoch):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode(testing, epoch)

        start_lives = self.ale.lives()  # This is just a constant of how many lives we start with
        # self.ale.lives() however is updated every time they lose a life. Where is this happening?

        action = self.agent.start_episode(self.get_observation())
        action = 0
        num_steps = 0
        t = 0
        while True:
            reward = self._step(self.min_action_set[action], testing)
            #self.terminal_lol = (self.death_ends_episode and not testing and
            #                     self.ale.lives() < start_lives)

            if self.time_step > self.steps_per_episode:
                self.terminal_lol = True

            # #################################
            # Changed this to determine when the game is over since my game does not have a defined ending point but
            # after some time steps
            terminal = self.ale.game_over() or self.terminal_lol
            num_steps += 1
            #if terminal or num_steps >= max_steps:
            if num_steps >= max_steps:
                mean_reward = self.agent.end_episode(reward, terminal)
                break

            action = self.agent.step(reward, self.get_observation())
            t += 1

        # print('EPISODE TRAINING REWARD', self.agent.episode_training_reward)
        return terminal, num_steps

    def get_observation(self):  # #####################################################################
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        # return self.resize_image(max_image)
        return max_image

    def resize_image(self, image):
        """ Appropriately resize a single image """

        if self.resize_method == 'crop':
            # resize keeping aspect ratio
            resize_height = int(round(
                float(self.height) * self.resized_width / self.width))

            resized = cv2.resize(image,
                                 (self.resized_width, resize_height),
                                 interpolation=cv2.INTER_LINEAR)

            # Crop the part we want
            crop_y_cutoff = resize_height - CROP_OFFSET - self.resized_height
            cropped = resized[crop_y_cutoff:
                              crop_y_cutoff + self.resized_height, :]

            return cropped
        elif self.resize_method == 'scale':
            return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError('Unrecognized image resize method.')
