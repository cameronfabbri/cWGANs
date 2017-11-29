# Conditional Wasserstein GANs

This is an implementation of conditional GANS using the Improved Wasserstein (WGAN-GP) method.
Currently I'm trying it out on multiple datasets, though Celeba has been the main target.

### Interpolation results
These results are obtained by choosing two values for z, (the same) random values for y, then
interpolating between the two z values. The images on the far left and far right sides are
the two initial values for z.

![i1](https://i.imgur.com/Ca6nRZt.png)

![i2](https://i.imgur.com/7sxwx1a.png)

![i3](https://i.imgur.com/PaDw1RV.png)
