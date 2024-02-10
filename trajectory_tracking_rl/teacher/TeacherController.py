import numpy as np
import copy
import pickle
from .Methods import RandomTeacher,AlpGMM,OracleTeacher

def param_vec_to_param_dict(param_env_bounds,param):
    param_dict = {}
    
    for i,(name,bound) in enumerate(param_env_bounds.items()):

        if len(bound) == 2:
            param_dict[name] = param[i]
        elif len(bound) == 3:
            param_dict[name] = param[i:i+bound[2]]

    return param_dict


class TeacherController:

    def __init__(self,param_env_bounds,env,args):

        self.param_env_bounds = param_env_bounds
        self.args = args

        min_param = []
        max_param = []

        for _,bound in param_env_bounds.items():

            if len(bound) == 2:

                min_param.append(bound[0])
                max_param.append(bound[1])
            
            elif len(bound) == 3:

                dim = bound[2]

                min_param.extend([bound[0]]*dim)
                max_param.extend([bound[1]]*dim)
            
            else:
                print("ERROR : PARAMS NOT IN FORMAT")

        if args.teach_alg == "random":
            self.task_generator = RandomTeacher.RandomTeacher(min_param,max_param,seed=0)
        elif args.teach_alg == "oracle":
            self.task_generator = OracleTeacher.OracleTeacher(min_param,max_param,[0.1]*4,seed=0)
        if args.teach_alg == "alp_gmm":
            self.task_generator = AlpGMM.ALPGMM(min_param,max_param,args,seed=0)

        self.env = env
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_param_bounds': list(self.param_env_bounds.items())}
            
            dump_dict = self.task_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def record_train_episode(self, reward, ep_len):
        self.env_train_rewards.append(reward)
        self.env_train_len.append(ep_len)
        if self.args.teach_alg != 'Oracle':
            reward = np.interp(reward, (-150, 350), (0, 1))
            self.env_train_norm_rewards.append(reward)
        self.task_generator.update(self.env_params_train[-1], reward)

    def generate_env_param(self):

        params = copy.copy(self.task_generator.generate_task())
        assert type(params[0]) == np.float32
        self.env_params_train.append(params)
        param_dict = param_vec_to_param_dict(self.param_env_bounds,params)
        self.env.set_env_variable(**param_dict)

if __name__ == "__main__":

    param = {}
    param["n_obstacles"] = [0,10]
    param["obs_center"] = [-6,6,3]

    teacher_ctr = TeacherController(param,env=None)

    teacher_ctr.generate_env_param()


    

        


