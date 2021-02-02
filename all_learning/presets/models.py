from all import nn  # note that torch.nn is also imported as nn
import torch


def fc_relu_q(env, hidden=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        nn.Linear(hidden, env.action_space.n),
    )


def dueling_fc_relu_q(env):
    return nn.Sequential(
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256), nn.ReLU(), nn.Linear(256, 1)
            ),
            nn.Sequential(
                nn.Linear(env.state_space.shape[0], 256),
                nn.ReLU(),
                nn.Linear(256, env.action_space.n),
            ),
        ),
    )


def fc_relu_features(env, hidden=64):
    return nn.Sequential(
        nn.Flatten(), nn.Linear(env.state_space.shape[0], hidden), nn.ReLU()
    )


def fc_value_head(hidden=64):
    return nn.Linear0(hidden, 1)


def fc_policy_head(env, hidden=64):
    return nn.Linear0(hidden, env.action_space.n)


def fc_relu_dist_q(env, hidden=64, atoms=51):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        nn.Linear0(hidden, env.action_space.n * atoms),
    )


def fc_relu_rainbow(env, hidden=64, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(env.state_space.shape[0], hidden),
        nn.ReLU(),
        nn.CategoricalDueling(
            nn.NoisyFactorizedLinear(hidden, atoms, sigma_init=sigma),
            nn.NoisyFactorizedLinear(
                hidden, env.action_space.n * atoms, init_scale=0.0, sigma_init=sigma
            ),
        ),
    )


OUT_DIM = {2: 61, 4: 57, 6: 53, 8: 49, 10: 45, 11: 43, 12: 41} # for image size 84*84

class PixelEncoder(nn.Module):
    """Convolutional encoder of pixel observations"""
    def __init__(self, obs_shape, feature_dim, num_layers=4, num_filters=32):
        super().__init__()
        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

    def forward_conv(self, obs):
        conv = torch.relu(self.convs[0](obs))
        #print(conv.shape)
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            #print(conv.shape)
            
        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs):
        h = self.forward_conv(obs)
        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        out = torch.tanh(h_norm)

        return out

def vision_q(env, feature_dim=50, conv_num_layers=4, conv_num_filters=32, q_hidden=64):
    #image_shape = [3,128,128]

    # [C,H,W]
    image_encoder = PixelEncoder(obs_shape=env.observation_space.shape, feature_dim=feature_dim, num_layers=conv_num_layers, num_filters=conv_num_filters)
    #image_encoder = PixelEncoder(obs_shape=image_shape, feature_dim=feature_dim, num_layers=conv_num_layers, num_filters=conv_num_filters)

    return nn.Sequential(image_encoder,  
        nn.Linear(feature_dim, q_hidden),
        nn.ReLU(),
        nn.Linear(q_hidden, env.action_space.n))
        #nn.Linear(q_hidden, 5))

if __name__ == "__main__":
    encoder = PixelEncoder(obs_shape=[3,128,128], feature_dim=50, num_layers=4, num_filters=32)
    vision_q = vision_q(None)
    print(vision_q) 
    data = torch.rand(64, 3, 128, 128) 
    output = vision_q.forward(data)
    print(output.shape)
                 
