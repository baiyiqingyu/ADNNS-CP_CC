from Models.High_OC import Policy_over_options_Learner
from Agents.AgentUtils import *


def mkdir(path):
    is_exists = os.path.exists(path)
    if not is_exists:
        os.makedirs(path)


class High_Level_Agent(object):

    def __init__(self, agent_ID, device, args):
        self.ag_idx = agent_ID
        self.HL_state_dim = args['state_dim_ideal']  # int
        self.HL_option_dim = args['option_dim']  # int
        self.device = device  # torch.device("cpu")
        self.seed = 1
        self.sample_count = 0  # 用于epsilon的衰减计数
        self.epsilon = 0.95
        self.epsilon_start = 0.95
        self.epsilon_end = 0.01
        self.epsilon_decay = 1000

        """
        High level learner
        """
        self.learner = Policy_over_options_Learner(self.HL_state_dim, 256, self.HL_option_dim, self.device, args)

    def init_training(self):
        self.learner.init_training()

    def add2replay_memory(self, state_old, option, reward, state_new, done, step_idx, epoch_idx=None):
        if self.learner.reply_buffer_type == "PER":
            self.learner.replay_buffer.push(state_old.flatten(), option, reward, state_new, done, step_idx, epoch_idx)
        else:
            self.learner.replay_buffer.push(state_old.flatten(), option, reward, state_new, done, step_idx)

    def get_option_during_train(self, state, current_option, step):
        step = torch.tensor(step, dtype=int, device=self.device).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        option = torch.LongTensor([current_option]).to(self.device).unsqueeze(0)
        terminate_prob = self.learner.Termination_Net(state, step).gather(1, option)
        terminate = torch.distributions.Bernoulli(terminate_prob).sample()
        # print(terminate_prob)
        # print("terminate_prob:", terminate_prob.item())

        if bool(terminate.item()):
            self.sample_count += 1
            self.epsilon = (self.epsilon_end + (self.epsilon_start - self.epsilon_end) *
                            math.exp(-1. * self.sample_count / self.epsilon_decay))
            if random.random() > self.epsilon:
                with torch.no_grad():
                    state_option_values = self.learner.Q_Net(state, step)
                    option = state_option_values.max(1)[1].item()
            else:
                option = random.randrange(self.HL_option_dim)
                # print(self.epsilon)


        else:
            option = current_option

        return option  # return scalar

    @torch.no_grad()
    def get_option_during_test(self, state, current_option, step):
        step = torch.tensor(step, dtype=int, device=self.device).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        option = torch.LongTensor([current_option]).to(self.device).unsqueeze(0)
        terminate_prob = self.learner.Termination_Net(state, step).gather(1, option)
        terminate = torch.distributions.Bernoulli(terminate_prob).sample()
        if bool(terminate.item()):
            state_option_values = self.learner.Q_Net(state, step)
            option = state_option_values.max(1)[1].item()

        else:
            option = current_option

        return option  # return scalar

    """
    Saving and Loading of Termination Policy Model
    """

    # load the saved action model
    def load_Termination_model(self, model_save_dir, file_str):
        DNN_model_save_dir = r"{}/Termination_models".format(model_save_dir)
        mkdir(DNN_model_save_dir)
        file_name = "agent_ID_" + str(self.ag_idx) + "_Termination" + "_" + str(file_str) + ".pt"
        DNN_model_save_path = os.path.join(DNN_model_save_dir, file_name)
        if os.path.exists(DNN_model_save_path):
            print('Load model')
            loaded_paras = torch.load(DNN_model_save_path, map_location=self.device)
            self.learner.Termination_Net.load_state_dict(loaded_paras)
        else:
            print("Error: No saved actor model of {}".format(DNN_model_save_path))
            raise ValueError("Error: No saved actor model of {}".format(DNN_model_save_path))

    # save the action model which is under training
    def save_Termination_model(self, model_save_dir, file_str):
        DNN_model_save_dir = r"{}/Termination_models".format(model_save_dir)
        mkdir(DNN_model_save_dir)
        file_name = "agent_ID_" + str(self.ag_idx) + "_Termination" + "_" + str(file_str) + ".pt"
        DNN_model_save_path = os.path.join(DNN_model_save_dir, file_name)
        model_state = copy.deepcopy(self.learner.Termination_Net.state_dict())
        torch.save(model_state, DNN_model_save_path)

    """
    Saving and Loading of Q_Net Model
    """

    # load the saved action model
    def load_QNet_model(self, model_save_dir, file_str):
        DNN_model_save_dir = r"{}/QNet_models".format(model_save_dir)
        mkdir(DNN_model_save_dir)
        file_name = "agent_ID_" + str(self.ag_idx) + "_QNet" + "_" + str(file_str) + ".pt"
        DNN_model_save_path = os.path.join(DNN_model_save_dir, file_name)
        if os.path.exists(DNN_model_save_path):
            print('Load model')
            loaded_paras = torch.load(DNN_model_save_path, map_location=self.device)
            self.learner.Q_Net.load_state_dict(loaded_paras)
        else:
            print("Error: No saved actor model of {}".format(DNN_model_save_path))
            raise ValueError("Error: No saved actor model of {}".format(DNN_model_save_path))

    # save the action model which is under training
    def save_QNet_model(self, model_save_dir, file_str):
        DNN_model_save_dir = r"{}/QNet_models".format(model_save_dir)
        mkdir(DNN_model_save_dir)
        file_name = "agent_ID_" + str(self.ag_idx) + "_QNet" + "_" + str(file_str) + ".pt"
        DNN_model_save_path = os.path.join(DNN_model_save_dir, file_name)
        model_state = copy.deepcopy(self.learner.Q_Net.state_dict())
        torch.save(model_state, DNN_model_save_path)
