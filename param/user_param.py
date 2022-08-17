class UserParam:
    def __init__(user_id, dataset):
        base_dir = 'fine-tune-dataset/'

        data_dir = base_dir + user_id
        target_dir = user_id + dataset
        direct_dir = data_dir + dataset

        sampling_rate = 22050