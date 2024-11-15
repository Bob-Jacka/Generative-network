from Entities.Discriminator import Discriminator
from Entities.Extensions import get_dataloader
from Entities.Generator import Generator

gen = Generator()
dis = Discriminator()

gen.set_name("generatorGAN")
dis.set_name("discriminatorGAN")

train_loader = get_dataloader(dis.training_set)

for i in range(50):
    g_loss = 0
    d_loss = 0
    for n, (real_samples, _) in enumerate(train_loader):
        loss_D = dis.train_D_on_real(real_samples)
        d_loss += loss_D
        loss_D = dis.train_D_on_fake(gen)
        d_loss += loss_D
        loss_G = gen.train_G()
        g_loss += loss_G
    g_loss = g_loss / n
    d_loss = d_loss / n
    if i % 10 == 9:
        print(f"at epoch {i + 1}, dloss: {d_loss}, gloss {g_loss}")
        gen.see_output()
