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


    keys_to_keep = [
        "adapter_layer", "adapter_dropout", "adapter_dim", "reversible_layer", "x1_factor",
        "x2_factor", "sum_factor", "finetune_output_layer"
    ]
    argparse_dict = vars(ckpt["args"])
    for k in argparse_dict.keys():
        if k not in keys_to_keep:
            del argparse_dict[k]
    print("---------- adapter args ---------------")
    print(argparse_dict)
    with open(os.path.join(adapter_dir, "args.json"), "w") as outfile:
        json.dump(argparse_dict, outfile)

if __name__ == "__main__":
    fire.Fire(main)

