import numpy as np
from scipy import stats, integrate

# Compute $\omega_\sigma(m)$ for a range of $m$s
def doubly_spatially_integrated_Gaussian_one_dim( sigma, support ):
    # Use expression (Owen, 1980) to resolve double spatial integration over Gaussian
    normal_zero_sigma = stats.norm( loc = 0, scale = sigma ) 
    return (
      (support + 1) * normal_zero_sigma.cdf( support + 1 )+
      (support - 1) * normal_zero_sigma.cdf( support - 1 )
    - (2 * support) * normal_zero_sigma.cdf( support     )
    + sigma**2 * (
                      normal_zero_sigma.pdf( support + 1 )+
                      normal_zero_sigma.pdf( support - 1 )
    - 2 *             normal_zero_sigma.pdf( support     )
     ) )

# Compute $\omega_\sigma(m)*\omega_\sigma(n)$ for a range of $m$s, $n$s and $\sigma$s
def stacked_doubly_spatially_integrated_Gaussian( sigmas, support ):
    # Initialize output, number of $m$s by number of $n$s by number of $\sigma$s
    stacked_output = np.empty( (support.size, support.size, sigmas.size) )
    for index in range( sigmas.size ):
        # Get one-dimensional for range
        auxiliar = doubly_spatially_integrated_Gaussian_one_dim( sigmas[index], support )
        # Outer product of two-dimensional in range x range
        stacked_output[...,index] = np.transpose( auxiliar )*auxiliar
    return stacked_output

# Once spatially integrated Gaussian, for simulating optical blur
def spatially_integrated_Gaussian( sigma = 2.279 ):
    support = np.array( [np.arange( -np.ceil( 3*sigma ), np.ceil( 3*sigma )+1, 1 )]  )
    normal_zero_sigma = norm( loc = 0, scale = sigma )
    auxiliar = ( normal_zero_sigma.cdf( support + .5 ) 
                    - normal_zero_sigma.cdf( support  -  .5 ) )
    return auxiliar * np.transpose( auxiliar )    

def obtain_discrete_kernels( sigma_limits ):
    # (!!) Extreme inefficiency, all filters are the same size, the maximum one (!!) See figure for smallest kernel below
    # Compute support for maximum sigma
    support = np.array( [np.arange( -np.ceil( 3*sigma_limits[-1] ), np.ceil( 3*sigma_limits[-1] )+1, 1 )], dtype = np.float32)
    # Initialize array
    kernels = np.empty( (support.size, support.size, sigma_limits.size-1), dtype = np.float32 )
    for index in range( sigma_limits.size-1 ):
        # Compute integral over \sigma
        kernels[...,index] = integrate.fixed_quad( stacked_doubly_spatially_integrated_Gaussian,
                                                   sigma_limits[index], sigma_limits[index+1],
                                                   args = (support,) )[0]
        kernels[...,index] = kernels[...,index] / np.sqrt( sigma_limits[index + 1] - sigma_limits[index] )
    return kernels
