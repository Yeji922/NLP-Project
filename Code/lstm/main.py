import sys
import os
import argparse
from Utils.train import train_pipeline
from Utils.predict import predict_text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    parser = argparse.ArgumentParser(description="Choose to train or predict")
    
    parser.add_argument('--opt', required=True, choices=['train', 'predict'],
                        help="Option to run: 'train' or 'predict'")
    parser.add_argument('--path', type=str, help="Path to the model, vocab, label_encoder for prediction/ Path to dataset for training")
    
    args = parser.parse_args()

    if args.opt == 'train':
        train_pipeline(args.path)
    elif args.opt == 'predict':
        model_path, vocab_path, label_encoder_path = args.path.split(",")
        predict_text(model_path.strip(), vocab_path.strip(), label_encoder_path.strip())



if __name__ == "__main__":
    main()


# saved_model/lstm_attention_model.pt, saved_model/vocab.pkl, saved_model/label_encoder.pkl