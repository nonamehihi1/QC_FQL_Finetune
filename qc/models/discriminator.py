import flax.linen as nn
import jax.numpy as jnp

class SuccessDiscriminator(nn.Module):
    hidden_dim: int = 256
    
    def setup(self):
        self.net = nn.Sequential([
            nn.Dense(self.hidden_dim), nn.relu,
            nn.Dense(self.hidden_dim), nn.relu,
            nn.Dense(1), nn.sigmoid
        ])
    
    def __call__(self, obs, act):
        x = jnp.concatenate([obs, act], axis=-1)
        return self.net(x)