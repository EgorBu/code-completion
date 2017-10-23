from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

LATENT_DIM = 256  # Default number of hidden neurons
MAX_ENCODER_LEN = 200  # Default length of input sequence
MAX_DECODER_LEN = 40  # Default length og output sequence


def prepare_seq2seq_model(token2ind=None, latent_dim=LATENT_DIM, optimizer="rmsprop",
                          encoder_seq_len=MAX_ENCODER_LEN, decoder_seq_len=MAX_DECODER_LEN):
    """
    Prepare seq2seq models for code.
    It includes: seq2seq model for training, encoder model & decoder model for inference step
    :param token2ind: token to index mapping (indices should start from 1)
    :param latent_dim: number of hidden neurons in LSTM
    :param optimizer: name of optimizer or optimizer with specific parameters
    :param encoder_seq_len: len of encoder sequence
    :param decoder_seq_len: len of decoder sequence
    :return: train_model, decode_sequence, ind2token: train_model - model for training,
                                                      decode_sequence - func to predict sequence
                                                      ind2token - index to token mapping
    """

    print("Number of unique tokens:", len(token2ind))
    print("Max sequence length for inputs:", encoder_seq_len)
    print("Max sequence length for outputs:", decoder_seq_len)

    # default padding value is 0 - so index should start from 1
    # and we need to keep additional space in
    num_tokens = len(token2ind) + 1
    emb_input_dim = num_tokens
    emb_output_dim = num_tokens
    emb_weights = np.eye(num_tokens)
    encoder_emb = Embedding(input_dim=emb_input_dim, output_dim=emb_output_dim,
                            weights=[emb_weights], trainable=False)

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    encoder_input_emb = encoder_emb(encoder_inputs)

    encoder = LSTM(latent_dim, return_state=True, name="encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_input_emb)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")
    decoder_in_emb = encoder_emb(decoder_inputs)
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_outputs, _, _ = decoder_lstm(decoder_in_emb, initial_state=encoder_states)
    decoder_dense = Dense(num_tokens, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    train_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print(train_model.summary())

    # compile
    train_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # Next: inference mode (sampling).

    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_in_emb,
                                                     initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    ind2token = dict((i, char) for char, i in token2ind.items())

    def decode_sequence(input_seq, start_ch="|", sep="", decoder_seq_len=decoder_seq_len):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0] = token2ind[start_ch]

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            # Sample a token
            sampled_token_index = output_tokens.argmax(axis=-1)[0, 0]
            sampled_char = ind2token.get(sampled_token_index, " ")
            decoded_sentence.append(sampled_char)

            # Exit condition: either hit max length
            # or find stop character.
            if len(decoded_sentence) > decoder_seq_len:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq[0, 0] = sampled_token_index

            # Update states
            states_value = [h, c]

        return sep.join(decoded_sentence)

    return train_model, decode_sequence, ind2token

