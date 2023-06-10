import json

import torch

from llama import ModelArgs, Tokenizer, Transformer


def RevLlama7B(args, **kwargs):

    llama_model_path = args.llama_model_path
    model_name = "7B"

    checkpoint = torch.load(llama_model_path + model_name + "/consolidated.00.pth", map_location="cpu")
    print(llama_model_path + model_name + "/consolidated.00.pth")

    with open(llama_model_path + model_name + "/params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=args.max_seq_len,
        max_batch_size=32,
        adapter_layer=args.adapter_layer,
        adapter_dropout=args.adapter_dropout,
        adapter_dim=args.adapter_dim,
        reversible_layer=args.reversible_layer,
        x1_factor=args.x1_factor,
        x2_factor=args.x2_factor,
        sum_factor=args.sum_factor,
        finetune_output_layer=args.finetune_output_layer,
        **params
    )
    tokenizer = Tokenizer(model_path=llama_model_path + "/tokenizer.model")

    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model_revllama = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model_revllama.load_state_dict(checkpoint, strict=False)

    for name, param in model_revllama.named_parameters():
        if ("adapter" not in name) and ("factor" not in name):
            param.requires_grad = False
        else:
            param.requires_grad = True
            param.data = param.data.float()

        if params.finetune_output_layer and ("output" in name):
            param.requires_grad = True
            param.data = param.data.float()

    """
    for name, param in model_revllama.layers[-1 * args.adapter_layer :].named_parameters():
        if "factor" in name or "adapter" in name:
            param.data = param.data.float()
            param.requires_grad = True
    """
    return model_revllama


# set recommended archs
RevLlama7B = RevLlama7B
