from Entities.Discriminator import Discriminator
from Entities.Generator import Generator

gen = Generator()
dis = Discriminator()

gen.set_name("generatorGAN")
dis.set_name("discriminatorGAN")

gen.test_epoch()