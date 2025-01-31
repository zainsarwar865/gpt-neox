import torch
import torch.nn as nn
from megatron.mpu.layers import _initialize_affine_weight_gpu



class ExpertEmbedding(nn.Module):
    def __init__(self, neox_args, init_method, output_layer_init_method):
        super().__init__()
        self.h_down = torch.nn.Parameter(
                torch.empty(neox_args.hidden_size + neox_args.intermediate_size,
                        neox_args.hypernetwork_bottleneck_dim,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
        )

        _initialize_affine_weight_gpu(
            self.h_down, init_method, partition_dim=0, stride=1)

        self.h_up = torch.nn.Parameter(
                torch.empty(neox_args.hidden_size + neox_args.intermediate_size,
                            neox_args.hypernetwork_bottleneck_dim,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
        )

        _initialize_affine_weight_gpu(
            self.h_down, init_method, partition_dim=0, stride=1)
        
    def forward(self, experts):
        # Get weights of the expert in expert_ids and do projections

        w2 = experts.w2.view(experts.experts_per_rank, -1, experts.hidden_size)
        w2_csum = torch.einsum('ijk->ij', w2)
        w2_rsum = torch.einsum('ijk->ik', w2)        
        w2_summed = torch.cat((w2_rsum, w2_csum), dim=1)
        w2_emb = torch.matmul(w2_summed, self.h_up)

        w1 = experts.w1.view(experts.experts_per_rank, experts.hidden_size, -1)
        w1_rsum = torch.einsum('ijk->ij', w1)
        w1_csum = torch.einsum('ijk->ik', w1)        
        w1_summed = torch.cat((w1_rsum, w1_csum), dim=1)
        w1_emb = torch.matmul(w1_summed, self.h_down)
        return w1_emb, w2_emb
        # return torch.cat((w1_emb, w2_emb), dim=1)