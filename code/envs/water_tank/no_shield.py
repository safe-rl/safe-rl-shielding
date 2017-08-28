class Shield:

    def __init__(self):
        self.water_level = 0
        self.switch_state = 0

    def tick(self, water_level, switch_state, action):
        return action
