from gym.spaces import Box
import numpy as np

class RandomTeacher:

    def __init__(self,min_param,max_param,seed=None):

        self.seed = seed
        if not seed:
            self.seed = np.random.randint(42,424242)
        np.random.seed(self.seed)
        
        self.space = Box(low= np.array(min_param),high=np.array(max_param),dtype=np.float32)

    def generate_task(self):

        return self.space.sample()
    
    def update(self, task, competence):
        pass

    def dump(self, dump_dict):
        return dump_dict
    
if __name__ == "__main__":

    min_param = [0,-6,-6,-6]
    max_param = [10,6,6,6]

    teacher = RandomTeacher(min_param,max_param)

    print(teacher.generate_task())

