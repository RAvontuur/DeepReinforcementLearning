from environment import ConnectFourEnvironment
from player import Player

from shutil import copyfile

from game import Game
from agent import Agent
from model import Residual_CNN

from settings import run_folder, run_archive_folder
import initialise



class PlayerAlphaZero(Player):

    def __init__(self):
        self.env_alpha_zero = Game()

        # If loading an existing neural network, copy the config file to root
        assert(initialise.INITIAL_RUN_NUMBER is not None)

        copyfile(
                run_archive_folder + self.env_alpha_zero.name + '/run' + str(initialise.INITIAL_RUN_NUMBER).zfill(4) + '/config.py',
                './config.py')

        import config

        ######## LOAD MODEL IF NECESSARY ########

        # create an untrained neural network objects from the config file
        best_NN = Residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + self.env_alpha_zero.grid_shape, self.env_alpha_zero.action_size,
                               config.HIDDEN_CNN_LAYERS)

        # If loading an existing neural netwrok, set the weights from that model
        assert(initialise.INITIAL_MODEL_VERSION is not None)
        best_player_version = initialise.INITIAL_MODEL_VERSION
        print('LOADING MODEL VERSION ' + str(initialise.INITIAL_MODEL_VERSION) + '...')
        m_tmp = best_NN.read(self.env_alpha_zero.name, initialise.INITIAL_RUN_NUMBER, best_player_version)
        best_NN.model.set_weights(m_tmp.get_weights())

        # copy the config file to the run folder
        copyfile('./config.py', run_folder + 'config.py')

        self.best_player = Agent('best_player', self.env_alpha_zero.state_size, self.env_alpha_zero.action_size, config.MCTS_SIMS, config.CPUCT, best_NN)

    def reset(self):
        self.env_alpha_zero.reset()
        self.best_player.mcts = None

    def play(self, env: ConnectFourEnvironment, untried_actions = None):
        assert(not env.terminated)

        if env.last_action is not None:
            action_opponent = env.last_action
            for row in range(6):
                if env.state[env.last_action][5 - row] != 0:
                    break
                action_opponent += 7
            state, value, done, _ = self.env_alpha_zero.step(action_opponent)
            assert (not done)

        action, pi, MCTS_value, NN_value = self.best_player.act(self.env_alpha_zero.gameState, 1)
        state, value, done, _ = self.env_alpha_zero.step(action)

        env_action = action % 7
        env_result = env.move(env_action)

        # print(self.render(self.env_alpha_zero))
        # print(env_result.display())

        assert(done == env_result.terminated)
        return env_result, env_action

    def render(self, env: Game):
        s = ""
        for r in range(6):
            for x in env.gameState.board[7 * r: (7 * r + 7)]:
                s = s + env.pieces[str(x)]
            s = s + "\n"
        return s