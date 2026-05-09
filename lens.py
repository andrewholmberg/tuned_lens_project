from pathlib import Path

import torch

class tuned_lens(torch.nn.Module):
    '''
    Class for tuned lens. main data structure is the unimbedding matrices.
    '''
    def __init__(self, model, device):
        '''
        Pass in model but don't save. just need some features to make the unimbedding matrix list.
        '''
        super().__init__()
        self.unembedding_matrices = torch.nn.ModuleList([
            torch.nn.Linear(model.cfg.d_model, model.cfg.d_vocab, bias=True).to(device)
            for _ in range(model.cfg.n_layers)
        ])

    def return_unembeddings(self, model, tokens):
            '''
            Return the unembeddings for each block. This is the main function of the tuned lens class.
            '''
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens)
            unembeddings = []
            for i in range(len(self.unembedding_matrices)):
                block_output = cache[f'blocks.{i}.hook_resid_post']
                unembedding_matrix = self.unembedding_matrices[i]
                unembedding = unembedding_matrix(block_output)
                unembeddings.append(unembedding)
            return unembeddings
    
    def save_lens(self, path):
        '''
        Save the unembedding matrices. This is the main function of the tuned lens class.
        '''
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "lens_type": "tuned_lens",
                "num_layers": len(self.unembedding_matrices),
                "unembedding_matrices_state_dict": {
                    key: value.detach().cpu()
                    for key, value in self.unembedding_matrices.state_dict().items()
                },
            },
            path,
        )
        return path

    def load_lens(self, path):
        '''
        Load the unembedding matrices. This is the main function of the tuned lens class.
        '''
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)

        if "unembedding_matrices_state_dict" in checkpoint:
            state_dict = checkpoint["unembedding_matrices_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if all(key.startswith("unembedding_matrices.") for key in state_dict):
            state_dict = {
                key.removeprefix("unembedding_matrices."): value
                for key, value in state_dict.items()
            }

        num_layers = len(self.unembedding_matrices)
        state_dict = {
            key: value.to(device)
            for key, value in state_dict.items()
            if not key.split(".", 1)[0].isdigit() or int(key.split(".", 1)[0]) < num_layers
        }
        self.unembedding_matrices.load_state_dict(state_dict)
        return self


class logit_lens(torch.nn.Module):
    '''
    class for logit lens. no trainable parameters, just uses the model's unembedding matrix to return the unembeddings for each block.
    '''
    def __init__(self):
        '''
        No structures needed here. I'll just have the user pass in the model and tokens to the return_unembeddings function and it will return the unembeddings for each block.
        '''
        super().__init__()

    def return_unembeddings(self, model, tokens):
        '''
        Return the unembeddings for each block. This is the main function of the logit lens class.
        '''
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        unembeddings = []
        for i in range(len(model.blocks)):
            block_output = cache[f'blocks.{i}.hook_resid_post']
            unembedding_matrix = model.unembed.W_U
            unembedding = unembedding_matrix(block_output)
            unembeddings.append(unembedding)
        return unembeddings
