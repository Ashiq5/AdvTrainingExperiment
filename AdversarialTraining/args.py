class Args:
    def __init__(self, attack_class_for_training=None, attack_class_for_testing=None,
                 dataset="imdb", model="lstm", data_files=None, type_of_file="csv",
                 epochs=50, allowed_labels=None, batch_size=64, max_length=512,
                 learning_rate=2e-5, output_dir=None, num_labels=2,
                 adversarial_samples_to_train=500, attack_period=50, num_attack_samples=500,
                 model_short_name="lstm",
                 at_model_prefix=None, orig_model_prefix=None):
        self.attack_class_for_training = attack_class_for_training
        self.attack_class_for_testing = attack_class_for_testing
        self.dataset = dataset
        self.model = model
        self.data_files = data_files
        self.type_of_file = type_of_file
        self.epochs = epochs
        self.allowed_labels = allowed_labels
        self.batch_size = batch_size
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.output_dir = output_dir
        self.num_labels = num_labels
        self.adversarial_training = attack_class_for_training is not None
        self.adversarial_samples_to_train = adversarial_samples_to_train
        self.attack_period = attack_period
        self.num_attack_samples = num_attack_samples
        self.model_short_name = model_short_name
        self.at_model_prefix = at_model_prefix
        self.orig_model_prefix = orig_model_prefix
