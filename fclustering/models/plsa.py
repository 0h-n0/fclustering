import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer import Predictive

from .fclustering_model import FClusteringModel


class PLSA(FClusteringModel):
    def __init__(self, num_topics: int, beta: float=1.) -> None:
        super().__init__()        
        self.num_topics = num_topics
        self.beta = beta

    def model(self, output=None):
        """ -> p(w, d) = sum_z p(z)p(w|z)p(d|z)
        """
        docs_loc = torch.zeros((output.shape[0], output.shape[1], 1), dtype=output.dtype, device=output.device)
        docs_scale = torch.ones((output.shape[0], output.shape[1], 1), dtype=output.dtype, device=output.device)
        words_loc = torch.zeros((output.shape[0], output.shape[2], 1), dtype=output.dtype, device=output.device)
        words_scale = torch.ones((output.shape[0], output.shape[2], 1), dtype=output.dtype, device=output.device)            
        p_docs = pyro.sample("docs", dist.Normal(docs_loc, docs_scale).to_event(2))
        p_words = pyro.sample("words", dist.Normal(words_loc, words_scale).to_event(2))
        z_loc = torch.zeros((output.shape[0], 1, self.num_topics), dtype=output.dtype, device=output.device)
        z_scale = torch.ones((output.shape[0], 1, self.num_topics), dtype=output.dtype, device=output.device)
        
        with pyro.plate("data", output.shape[0]):
            p_z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(2))
            p_docs_z = torch.bmm(p_docs, p_z)
            p_words_z = torch.bmm(p_words, p_z).transpose(1, 2)
            mean = torch.bmm(p_docs_z, p_words_z)

    def get_guide(self):
        return pyro.infer.autoguide.AutoNormal(self.model)

    def train(self, data):        
        adam = pyro.optim.Adam({"lr": 0.02})  # Consider decreasing learning rate.
        elbo = pyro.infer.Trace_ELBO()
        self.guide = self.get_guide()
        svi = pyro.infer.SVI(self.model, self.guide, adam, elbo)
        losses = []
        for step in range(2000):  # Consider running for more steps.
            loss = svi.step(data)
            losses.append(loss)
            if step % 100 == 0:
                print("Elbo loss: {}".format(loss))

    def predict(self):
        self.guide.requires_grad_(False)
        z = pyro.get_param_store()["AutoNormal.locs.latent"]
        docs = pyro.get_param_store()["AutoNormal.locs.docs"]        
        words = pyro.get_param_store()["AutoNormal.locs.words"]   
        print(z, docs, words)     
        docs_z = torch.bmm(docs, z)
        words_z = torch.bmm(words, z)
        print(docs)
        print(docs_z)
        print(words)
        print(words_z.T)
        

        
if __name__ == "__main__":
    pass