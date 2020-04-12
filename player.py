from typing import List
from environment import ConnectFourEnvironment


class Player:

    def reset(self):
        pass

    def action_values(self, env: ConnectFourEnvironment) -> List[float]:
        pass

    def play(self, env: ConnectFourEnvironment) -> [ConnectFourEnvironment, int]:
        pass

    def analyzed_result(self):
        return None
