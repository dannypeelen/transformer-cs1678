import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm  # TODO use this one if running on jupyter notebook (colab)
from tqdm import tqdm  # use this one if running on terminal
from transformer import Transformer
from helper import model_name, val_loss
from numba import cuda

def get_batch(D, context_size, batch_size):
    """
    Get a batch of data from the dataset.

    Parameters:
    D: torch.tensor of shape (L,) and dtype torch.long (int)
        The dataset
    context_size: int
        The number of characters to use as context
    batch_size: int
        The number of sampled subsequences for the mini-batch

    Returns:
    torch.tensor of shape (context_size, batch_size)
        The input data for the model
    torch.tensor of shape (context_size, batch_size)
        The target data for the model
    """
    n = len(D)
    max_index = n - context_size - 1
    j = torch.randint(0, max_index, (batch_size,)) # specifies how to format the random number as a tensor

    x = torch.stack([D[i: i + context_size] for i in j], dim=1) # TODO fill in the code to get the input data
    y = torch.stack([D[i+1: i+ context_size + 1] for i in j], dim=1) # TODO fill in the code to get the target data
    # TODO make sure to test this function to make sure the sequences look correct. 
    # If you get the wrong shape of the data or order the data incorrectly the model won't learn what you want it to and will struggle.
    # a simple way to test this is to convert the integer sequences back to text and print them out.
    return x, y

def train(Dtrain, Dval, model, optimizer, criterion, context_size, batch_size, num_epochs, save_freq=10, save_name="transformer"):
    """
    Train the model on the data. Save the model every save_freq epochs.

    Parameters:
    Dtrain: torch.tensor of shape (Ltrain,)
        The training data
    Dval: torch.tensor of shape (Lval,)
        The validation data
    model: nn.Module
        The model to train
    optimizer: torch.optim.Optimizer
        The optimizer to use
    criterion: callable function
        The loss function to compare the model output to the target
    context_size: int
        The number of characters to use as context
    batch_size: int
        The number of samples in each batch
    num_epochs: int
        The number of epochs to train for
    save_freq: int
        The frequency with which to save the model
    save_name: str
        The name to use when saving the model
    """
    # TODO read over this function to understand how to train the model.
    tlosses = []
    vlosses = []
    updates_per_epoch = len(Dtrain)//(batch_size*context_size)
    # prep the validation data
    even_eval = len(Dval)//(context_size+1)
    Dval = Dval[:even_eval*(context_size+1)].reshape(-1, context_size+1).T

    # create the mask for the transformer
    mask = torch.nn.Transformer.generate_square_subsequent_mask(context_size)  # create the mask for the transformer which is always the same because we sample the same length sequences
    mask = mask.to(Dtrain.device)  # move the mask to the same device as the data

    for epoch in tqdm(range(num_epochs)):  # print out progress bar
        vloss = val_loss(Dval, model, criterion, vocab_size, mask)
        vlosses.append(vloss)
        model.train() # set the model to training mode
        total_loss = 0
        print(f"Epoch {epoch+1}: validation loss: {vloss}")
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), f"weights/{save_name}_{epoch}.pt")  # saves model weights but not the model itself. This means you must create the exact same model and load the weights into it to reuse the model.
        for _ in (range(updates_per_epoch)):  # loop over the dataset
            x, y = get_batch(Dtrain, context_size, batch_size)
            optimizer.zero_grad()
            y_pred = model(x, mask) # forward pass through the model
            loss = criterion(y_pred.view(-1, vocab_size), y.view(-1))  # compute the loss reshaping the model outputs to be shape (seq_len*batch_size, vocab_size) and the targets to be shape (seq_len*batch_size)
            loss.backward()  # backpropagate the loss
            optimizer.step()  # update the model weights
            total_loss += loss.item()
        tlosses.append(total_loss/updates_per_epoch)
        print(f"Epoch {epoch+1}, train loss: {total_loss/updates_per_epoch}")
    return model, tlosses, vlosses


def get_data():
    # use !wget to download the file if on collab
    # !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    # download the file yourself from github if running locally
    fname = "input.txt"
    with open(fname, 'r', encoding='utf-8') as f:
        text = f.read()
    chars = set(sorted(text)) # TODO create a list of characters. Probably best to sort them if you want to have a consistent mapping
    vocab_size = len(chars)  # number of unique characters
    print(f"vocab size: {vocab_size}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}  # TODO create a dictionary mapping characters to integers 
    itos = {i: ch for i, ch in enumerate(chars)}  # TODO create a dictionary mapping integers to characters
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


    data = torch.tensor(encode(text), dtype=torch.long)  # convert the text to integers
    dtrain = data[:int(0.9* len(data))]  # 90% of the data for training
    dval = data[int(0.9*len(data)):]    # 10% of the data for validation
    return dtrain, dval, vocab_size, encode, decode

def train_main(dtrain, dval, vocab_size, params):
    """
    Train the model with the given parameters.
    
    Parameters:
    dtrain: torch.tensor of shape (Ltrain,)
        The training data
    dval: torch.tensor of shape (Lval,)
        The validation data
    vocab_size: int
        The number of characters in the vocabulary
    params: dict
        A dictionary of parameters for the model

    Returns:
    nn.Module
        The trained model
    list of float
        The training losses
    list of float
        The validation losses
        
    """
    # TODO read through this function to understand how it trains the model
    if params['pos_enc']:
        save_name = f"transformer_e{params['embed_size']}_cs{params['context_size']}_nl{params['num_layers']}_nh{params['nhead']}"
    else:
        save_name = f"transformer_e{params['embed_size']}_cs{params['context_size']}_nl{params['num_layers']}_nh{params['nhead']}_nope"
    # create the model
    model = Transformer(vocab_size, embed_size=params['embed_size'], num_layers=params['num_layers'],
                        nhead=params['nhead'], layer_width=params['layer_width'], max_len=params['max_len'], N=params['N'], pos_enc=params['pos_enc'])

    if torch.cuda.is_available():  # move the model to the GPU if available
        dev = torch.device("cuda")
        print("cuda")
    else:
        dev = torch.device("cpu")
        print("cpu")

    dtrain = dtrain.to(dev) # move the data to the device
    dval = dval.to(dev)  # move the data to the device
    model = model.to(dev)  # move the model to the device
    stepsize = 1e-4  # learning rate for Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=stepsize)  # use Adam optimizer
    criterion = nn.CrossEntropyLoss().to(dev)  # use cross entropy loss

    model, tlosses, vlosses = train(dtrain, dval, model, optimizer, criterion, 
                                    params['context_size'], params['batch_size'], params['num_epochs'], 
                                    save_freq=5, save_name=save_name)  # train the model
    torch.save(model.state_dict(), f"weights/{save_name}_{params['num_epochs']}.pt")  # save the model weights at the end of training
    
    # plot the training and validation losses
    plt.figure()
    plt.plot(tlosses, label='train')
    plt.plot(vlosses, label='val')
    plt.legend()
    plt.savefig(f'{save_name}_losses.png')
    
    return model, tlosses, vlosses

from helper import generate_text, plot_attention

def answer_prompts(prompt, model, model_nope, file_name, cs, encode, decode):
    text = generate_text(model, prompt, 500, cs, encode, decode)  # TODO uncomment this line to generate text
    print(text)
    print("============WITHOUT POS ENCODING============")
    text2 = generate_text(model_nope, prompt, 500, cs, encode, decode)  # TODO uncomment this line to generate text
    print(text2)

    with open(file_name, "w") as f:
        f.write(text)
        f.write("\n==========WITHOUT POS ENCODING============\n")
        f.write(text2)

    return
    
if __name__ == "__main__":
    dtrain, dval, vocab_size, encode, decode = get_data()
    # change these hyperparameters. This should be the minimal sized model you train to report your results. 
    # to debug use 2 layers and smaller values for the other parameters
    # you can also run with larger values to see how the model performs
    esize=256  # embedding size
    cs=128  # context size
    nl=6 # number of transformer layers
    nh=8  # number of heads in the multiheadattention models
    width = esize*4  # width of the feedforward network
    pos_enc = True  # to use positional encoder or not
    params = {
        'context_size': cs,
        'batch_size': 64,
        'num_epochs': 60,  
        'embed_size': esize,
        'num_layers': nl,
        'nhead': nh,
        'layer_width': width,
        'max_len': 1024+32,  # max_len should be the sum of the prompt size and the maximum length of the generated text. 
        'N': 512.0,  # N should be longer than the longest sequence you will use. Probably don't need to change this. 
        'pos_enc': pos_enc  
    }

    ex1, ex2 = get_batch(dtrain, params['context_size'], params['batch_size'])  # get a batch of data to test the get_batch function
    assert ex1.shape == torch.Size([params['context_size'], params['batch_size']]), "returns incorrect shape"
    assert ex2.shape == torch.Size([params['context_size'], params['batch_size']]), "returns incorrect shape"

    model, tlosses, vlosses = train_main(dtrain, dval, vocab_size, params)  # TODO uncomment this line to train the model
    params['pos_enc'] = False
    model_nope, tlosses_nope, vlosses_nope = train_main(dtrain, dval, vocab_size, params) 


    # prep the eval data to select best saved model
    even_eval = len(dval)//(cs+1)
    dval = dval[:even_eval*(cs+1)].reshape(-1, cs+1).T
    prompt = "bat cat"  # prompt to examine the attention weights. TODO provide plots with this prompt in your write up. However you should change this to see how attention changes with different prompts.
    use_big_model = False  # set to True to use the big model that was provided. This will load the model from the file model_posenc_True.pth or model_posenc_False.pth
    #plot_attention(prompt, dval, vocab_size, esize, cs, nl, nh, width, encode, max_epochs=60, use_big_model=use_big_model)  # TODO uncomment this line to plot the attention weights
    
    # model = Transformer(vocab_size, embed_size=params['embed_size'], num_layers=params['num_layers'],
    #                     nhead=params['nhead'], layer_width=params['layer_width'], max_len=params['max_len'], N=params['N'], pos_enc=params['pos_enc'])  # TODO specify which model you want to use for text generation
    # params['pos_enc'] = False
    # model_nope = Transformer(vocab_size, embed_size=params['embed_size'], num_layers=params['num_layers'],
    #                     nhead=params['nhead'], layer_width=params['layer_width'], max_len=params['max_len'], N=params['N'], pos_enc=params['pos_enc']) 
    
    # model.load_state_dict(torch.load(f"weights/transformer_e256_cs128_nl6_nh8_80.pt", weights_only=True, map_location='cpu'))  # TODO load the model you want to use for text generation. This should be the model you trained above.
    # model_nope.load_state_dict(torch.load(f"weights/transformer_e256_cs128_nl6_nh8_nope_75.pt", weights_only=True, map_location='cpu'))  # TODO load the model you want to use for text generation. This should be the model you trained above.

    prompts = ["Surely, you must not want milk in your tea.", "A horse! A horse! A kingdom for my horse!", "But sir, the Big Mac is supposedly five dollars!", "No tomfoolery allowed here!"] # TODO: place your prompt to start the text generation here
    #print("With positional encoding...train loss: " + str(tlosses[-1]) + " validation loss: "  + str(vlosses[-1]))
    #print("Without positional encoding...train loss: " + str(tlosses_nope[-1]) + " validation loss: "  + str(vlosses_nope[-1]))
    num = 0
    for prompt in prompts:
        file_name = "generated_text_" + str(num) + ".txt"
        num += 1
        print("Prompt: ", prompt)
        answer_prompts(prompt, model, model_nope, file_name, cs, encode, decode)  # TODO uncomment this line to generate text
        print("\n=========================================\n")
    