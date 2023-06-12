import torch
import fire
import os
import json

def main(
    ckpt_path: str,
    adapter_dir: str,
    finetune_output_layer: bool = False,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    new_model = dict()

    for k, v in ckpt["model"].items():
        if ("adapter" in k) or ("factor" in k):
            new_model[k] = v
        if finetune_output_layer and ("output" in k):
            new_model[k] = v

    print("---------- adapter keys ---------------")
    print(new_model.keys())
    torch.save(new_model, os.path.join(adapter_dir, "adapter_parameters.pth"))


    keys_to_save = [
        "adapter_layer", "adapter_dropout", "adapter_dim", "reversible_layer", "x1_factor",
        "x2_factor", "sum_factor", "finetune_output_layer"
    ]
    argparse_dict = vars(ckpt["args"])
    args_to_save = {}
    for k, v in argparse_dict.items():
        if k in keys_to_save:
            args_to_save[k] = v
    print("---------- adapter args ---------------")
    print(args_to_save)
    with open(os.path.join(adapter_dir, "args.json"), "w") as outfile:
        json.dump(args_to_save, outfile)

if __name__ == "__main__":
    fire.Fire(main)

