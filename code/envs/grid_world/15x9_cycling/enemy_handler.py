class EnemyHandler:
    def __init__(self):
        self.positions = [(4,3)]
    
    def getEnemyPositions(self):
        return self.positions
        
    def update(self, good):
        for i in range(len(self.positions)):
            next = self.move(self.positions[i], good)
            if next != good:
                self.positions[i] = next
        
    def move(self, e, good):
        if e[1] == 3:
            if e == (9,3): return (9,4)
            else: return (e[0] + 1, 3)
        elif e == (9,4): return (9,5)
        elif e[1] == 5:
            if e == (5,5): return (5,4)
            else: return (e[0] - 1, 5)
        else: return (5,3)
    
    def move_reverse(self, e, good):
        if e[1] == 3:
            if e == (5,3): return (5,4)
            else: return (e[0] - 1, 3)
        elif e == (5,4): return (5,5)
        elif e[1] == 5:
            if e == (9,5): return (9,4)
            else: return (e[0] + 1, 5)
        else: return (9,3)
        