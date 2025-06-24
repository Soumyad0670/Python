class Door:
    def __init__(self, state):
        self.state = state
    def toggle_state(self):
        self.state = not self.state
open_door = 0
close_door = 0
doors = [None] + [Door(False) for _ in range(100)]  
for person in range(1, 101):
    for n in range(person, 101, person):
        doors[n].toggle_state()
for i in range(1, 101):
    if doors[i].state:
        open_door += 1
    else:
        close_door += 1
print("Open Doors:", open_door)
print("Close Doors:", close_door)

