class TrafficLight:
    def __init__(self):
        self.light = 'green'
        self.timer = 0
    
    def change_light(self):
        if self.light == 'green':
            self.light = 'red'
        else:
            self.light = 'green'
    
    def change_timer(self, speed='normal'):
        if speed == 'faster':
            self.timer += 3
        elif speed == 'slower':
            self.timer +=  1
        else:
            self.timer += 2
        
        if self.timer > 180:
            self.timer = 0
            self.change_light()

    

    