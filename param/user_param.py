class UserParam:
    def __init__(self, user_id, dataset):
        self.base_dir = 'fine-tune-dataset/'
        self.user_id = user_id + '/'
        self.dataset = dataset

        self.data_dir = self.base_dir + self.user_id
        self.target_dir = self.user_id + self.dataset
        self.direct_dir = self.data_dir + self.dataset

        self.sampling_rate = 22050