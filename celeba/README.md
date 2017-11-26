### Celeba File Details

`celeba_cgan.py`

Conditional GAN that given noise and and attribute vector generates images. This does NOT
create the dataset. We use the model this saves to create a dataset.


`generate_celeba.py`

Loads up the model saved by `celeba_cgan.py` and generates a number of images and their 
latent z variables which were randomly sampled.


`enc_z.py`

Loads up the images and latent z and trains an autoencoder to map from x->z, such that
we now have a mapping from x->z to use with the cgan

`gen_test.py`

Loads up the autoencoder that we trained and generates pairs of images and latent z
variables. These will be loaded up by icgan.


`icgan.py`

Loads up the generated test data so that we have an original image, its reconstruction,
and then swap the attribute to generate a new image.

