class TrafficLight:
    def __init__(self):
        self.light = 'green'
        self.timer = 60
    
    def change_light(self):
        if self.light == 'green':
            self.light = 'red'
        if self.light == 'red':
            self.light = 'green'
    

    