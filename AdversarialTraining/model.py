from textattack.models.helpers.lstm_for_classification import LSTMForClassification
from textattack.models.helpers.word_cnn_for_classification import WordCNNForClassification
from textattack.models.wrappers import PyTorchModelWrapper


def lstm_model(args):
    model = LSTMForClassification(
                max_seq_length=args.max_length,
                num_labels=args.num_labels,
                emb_layer_trainable=False,)
    model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    return model_wrapper


def cnn_model(args):
    model = WordCNNForClassification(
        max_seq_length=args.max_length,
        num_labels=args.num_labels,
        emb_layer_trainable=False
    )
    model_wrapper = PyTorchModelWrapper(model, model.tokenizer)
    return model_wrapper


def bert_model():
    pass
