First run `mnist_cgan.py`, which trains a conditional GAN to generate mnist digits conditioned
on a digit label.

Second, run `generate_mnist.py` which generates a bunch of images with their z vector that was
use in the generator to generate them. These pairs are to be used to train the encoder.

Third, run `enc_z.py` which trains the encoder to encode an image to its latent variable z.

Fourth, run `gen_test.py`, which will generate a bunch of latent z variables from REAL images.
These will be used in `icgan.py` with their attributes swapped.



