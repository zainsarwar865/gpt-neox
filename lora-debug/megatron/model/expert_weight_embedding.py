import torch
import torch.nn as nn
from megatron.mpu.layers import _initialize_affine_weight_gpu



class ExpertEmbedding(nn.Module):
    def __init__(self, neox_args, init_method, output_layer_init_method):
        super().__init__()
        #self.h_down = nn.Linear(neox_args.hidden_size + neox_args.intermediate_size, neox_args.hypernetwork_bottleneck_dim)
        #self.h_up = nn.Linear(neox_args.hypernetwork_bottleneck_dim, neox_args.hidden_size)

        self.h_down = torch.nn.Parameter(
                torch.empty(neox_args.hidden_size + neox_args.intermediate_size,
                        neox_args.hypernetwork_bottleneck_dim,
                        device=torch.cuda.current_device(),
                        dtype=neox_args.params_dtype,
                    )
        )

        _initialize_affine_weight_gpu(
            self.h_down, init_method, partition_dim=0, stride=1)

        # self.h_up = torch.nn.Parameter(
        #         torch.empty(neox_args.hypernetwork_bottleneck_dim,
        #                     neox_args.hidden_size,
        #                     device=torch.cuda.current_device(),
        #                     dtype=neox_args.params_dtype,
        #             )
        # )

        # _initialize_affine_weight_gpu(
        #     self.h_up, output_layer_init_method, partition_dim=0, stride=1
        # )


    # def forward(self, tokens, expert_ids, experts):
    #     # Get weights of the expert in expert_ids and do projections

    #     w1 = experts.w1.view(experts.experts_per_rank, experts.hidden_size, -1)
    #     w1_rsum = torch.einsum('ijk->ij', w1)
    #     w1_csum = torch.einsum('ijk->ik', w1)        
    #     w1_summed = torch.cat((w1_rsum, w1_csum), dim=1)
    #     w1_summed_indexed = w1_summed[expert_ids]
    #     #w1_indexed = w1[expert_ids] # [num_tokens, hid_dim, inter_dim]
    #     #w1_rsum = torch.einsum('ijk->ij', w1_indexed)
    #     #w1_csum = torch.einsum('ijk->ik', w1_indexed)
    #     hidden_reps = torch.cat((tokens, w1_summed_indexed), dim=1)
    #     #hidden_reps = torch.bmm(hidden_reps, self.h_down)
    #     #hidden_reps = torch.bmm(hidden_reps, self.h_up)
    #     hidden_reps = self.h_up(self.h_down(hidden_reps)) # Activation function?
    #     return hidden_reps
        
    def forward(self, experts):
        # Get weights of the expert in expert_ids and do projections

        w1 = experts.w1.view(experts.experts_per_rank, experts.hidden_size, -1)
        w1_rsum = torch.einsum('ijk->ij', w1)
        w1_csum = torch.einsum('ijk->ik', w1)        
        w1_summed = torch.cat((w1_rsum, w1_csum), dim=1)
        #w1_summed_indexed = w1_summed[expert_ids]
        #w1_indexed = w1[expert_ids] # [num_tokens, hid_dim, inter_dim]
        #w1_rsum = torch.einsum('ijk->ij', w1_indexed)
        #w1_csum = torch.einsum('ijk->ik', w1_indexed)
        #hidden_reps = torch.cat((tokens, w1_summed_indexed), dim=1)
        #hidden_reps = torch.bmm(hidden_reps, self.h_down)
        #hidden_reps = torch.bmm(hidden_reps, self.h_up)
        #hidden_reps = self.h_up(self.h_down(hidden_reps)) # Activation function?
        return torch.matmul(w1_summed, self.h_down)
        #hidden_reps = self.h_down(w1_summed) # Activation function?
        #return hidden_reps
        







