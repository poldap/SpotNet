import numpy as np
import tensorflow as tf
from scipy import integrate,special,linalg
from scipy.stats import norm
from .ker import spatially_integrated_Gaussian

# Generate SDRs with square pulse shapes
def get_SDRs( nrof_cells = 750,        nrof_time_points = 1000, seconds_experiment_length = 8*3600,
              earliest_start = 1*3600, latest_end = 6*3600,    smallest_ratio = 0.5 ):
    
    # Get integers for start and end points for each SDR and ensure they are not the same
    while True:
        start_and_end_times = np.sort( 
                                  np.random.randint( 
                                      low  = (nrof_time_points-1) * (earliest_start/seconds_experiment_length),
                                      high = (nrof_time_points-1) * ( latest_end   /seconds_experiment_length), 
                                      size = (nrof_cells, 2) ),
                              axis = 1 )
        if ~np.any( start_and_end_times[:,0] == start_and_end_times[:,1] ):
            break
    
    # Fill SDRs between start and end with a constant random value corresponding to total secretion between
    # a maximum (arbitrarily fixed to nrof_time_points) and the value given by the smallest considered ratio
    sdrs = np.zeros( (nrof_cells,nrof_time_points) )
    # Fill it up according to given values
    for idx, _ in enumerate( sdrs ):
        sdrs[idx,start_and_end_times[idx,0]:(start_and_end_times[idx,1]+1)] = (
            np.random.uniform( low = smallest_ratio*nrof_time_points, high = nrof_time_points ) / (
            start_and_end_times[idx,1] - start_and_end_times[idx,0] ) )
    
    # Compute limits of each of the intervals of the step function
    t_limits = np.arange( nrof_time_points + 1 )*seconds_experiment_length/nrof_time_points
    
    return ( sdrs, t_limits )


# Transform SDRs to delay-dependant Vs
def transform_SDRs_to_Vs( sdrs, t_limits, 
    diffusion_constant = 3e-12, adsorption_constant = 6e-9, desorption_constant = 1e-4, 
    nrof_rebinding_events_truncate = 20, provide_varphi = False ):
    
    # Enforce input shape (vertical vectors for time component)
    t_limits = np.reshape( t_limits, (t_limits.size, 1) )
    
    # Extract implicit inputs
    nrof_cells, nrof_time_points = sdrs.shape
    t_centers = 0.5*(t_limits[:-1] + t_limits[1:])
    
    ## Compute discretization of $\phi(\tau)$
    # Compute discretization of $\phi(\tau)$, analytical part
    phi_ini = ( 2*adsorption_constant*(np.sqrt( t_limits[1:] ) - np.sqrt( t_limits[:-1] ))/
            np.sqrt( np.pi*diffusion_constant ) )
    # Compute discretization of $\phi(\tau)$, numerical part
    erfcx_term = lambda tau: (adsorption_constant**2 / diffusion_constant) * (
                special.erfcx( adsorption_constant * np.sqrt( tau / diffusion_constant ) ))
    for idx, _ in enumerate( phi_ini ):
        phi_ini[idx] -= integrate.quad( erfcx_term, t_limits[idx], t_limits[idx+1] )[0] 
    
    ## Prepare for recursive computation of v (and optionally, \varphi)
    # Invert time in SDRs to prepare approximation of temporal integral
    sdrs = np.fliplr( sdrs );
    
    # Initialize output variables (obtained by accumulation of terms)
    
    # Output variable, containing all the $v_c(\tau,T)$s
    vs = np.zeros( (nrof_cells, nrof_time_points) )
    # Intermediate variable (not needed), $\varphi(\tau,t)$
    if provide_varphi:
        varphi = np.zeros( nrof_time_points, nrof_time_points )
    
    # Helping function for Poisson PMFs
    poisson_pmf = lambda val, lam: np.exp( -lam ) * lam**val / special.factorial( val )
    
    ## Recursively approximate v
    for rebind in range( nrof_rebinding_events_truncate ):
        
        if rebind == 0:
            # For a single rebind, load discretized $\phi(\tau)$
            phi_rebind = phi_ini
        else:
            # For the subsequent rebinds, perform discretized convolution to approximate $\phi^j(\tau)$
            phi_rebind = np.expand_dims(
                            np.convolve( phi_rebind[:,0], phi_ini[:,0] )
                            , 1 )
            # Clip for only the smallest times in free motion (indicator in formulas).
            # Note that doing this does not interfere with the computation of the 0:nrof_time_points section of higher
            # convolutional powers, and avoids unnecessary computations
            phi_rebind = phi_rebind[:nrof_time_points]
        
        # Array representing the different values taken by the Poisson distribution
        # expression when one changes \tau and \eta, dim 0 is \tau and dim 1 is \eta
        poisson_with_indicator_rebind = linalg.triu( linalg.toeplitz( 
                poisson_pmf( rebind, 
                            desorption_constant * np.reshape( t_centers, (1, t_centers.size) ) 
                           ) ) )  
        
        # Update on $\varphi(\tau,t)$
        if provide_varphi:
            varphi += phi_rebind * poisson_with_indicator_rebind
        
        # Update on each of the $v_c(\tau,T)$
        vs += ( np.sum( 
                 np.expand_dims( sdrs, 2 ) * 
                 np.expand_dims( poisson_with_indicator_rebind.swapaxes( 0, 1 ), 0 )    
               , 1, keepdims = True ) *  np.expand_dims( phi_rebind.swapaxes( 0, 1 ), 0 ) 
             ).swapaxes( 1, 2 ).squeeze( axis = 2 )
    
    if provide_varphi:
        return ( vs, varphi )
    return vs


# Transform delay-dependant Vs into PSDRs
def transform_Vs_to_PSDRs( vs, t_limits, nrof_sigma_centers = 30, 
                           diffusion_constant = 3e-12, pixel_length = 6.45e-6,
                           representation_as_papers = True ):
    
    # Compute existing boundaries in the \sigma domain
    time_limits_to_sigma = np.sqrt( 2 * diffusion_constant * t_limits ) / pixel_length
    
    ## Generate ideal grid
    # If the user requirement is impossible (we don't have enough resolution)
    if time_limits_to_sigma[1] > time_limits_to_sigma[-1] / nrof_sigma_centers:
        # Compute grid with the most points (for geeks,sqrt is expansive at the 
        # beginning and compressive at the end)
        sigma_limits_ideal = np.arange( time_limits_to_sigma[0 ], 
                                        time_limits_to_sigma[-1],
                                        time_limits_to_sigma[1 ] )
        sigma_limits_ideal = np.append( sigma_limits_ideal, time_limits_to_sigma[-1] )
        # Issue warning (TODO: Change to proper warning)
        print( "nrof_sigma_points unfeasible, " + str( sigma_limits_ideal.size ) + 
               " points used instead" )
    else:
        sigma_limits_ideal = np.linspace( time_limits_to_sigma[0 ], 
                                          time_limits_to_sigma[-1],
                                          num = nrof_sigma_centers + 1 )
    
    ## Find closest possible approximation
    # Compute distance (broadcasting)
    distance_matrix = np.absolute( np.expand_dims( 
                            sigma_limits_ideal,
                      axis = 1 )
                      - np.reshape( 
                            time_limits_to_sigma, 
                      ( 1, time_limits_to_sigma.size ) ) )
    # Find minimum distance element in existing limits for each ideal \sigma limit
    # and build real \sigma limits array
    sigma_limits_indices = np.argmin( distance_matrix, axis = 1 )
    sigma_limits_real = time_limits_to_sigma[sigma_limits_indices]
    
    ## Create PSDRs 
    # Initialize. Note: (sigma_limits_real.size - 1 should be nrof_sigma_centers in normal cases)
    psdrs = np.empty( ( vs.shape[0], sigma_limits_real.size - 1 ) )
    for idx in range( sigma_limits_real.size - 1 ):
        # Integrate over intervals and divide by length
        psdrs[:,idx] = np.sum( 
            vs[:,sigma_limits_indices[idx]:sigma_limits_indices[idx+1]], axis = 1 )
        if representation_as_papers:
            psdrs[:,idx] = psdrs[:,idx] / np.sqrt(
                    sigma_limits_real[idx+1] - sigma_limits_real[idx] )
        else:
            psdrs[:,idx] = psdrs[:,idx] / (
                sigma_limits_real[idx+1] - sigma_limits_real[idx] ) 
    
    return ( psdrs, sigma_limits_real )


# Generate a random FluoroSpot experiment
def generate_experiment( nrof_cells = 750, nrof_pixels = 512 ):
    sdrs, t_lims = get_SDRs( nrof_cells = nrof_cells )
    vs = transform_SDRs_to_Vs( sdrs, t_lims )
    psdrs, sigma_lims = transform_Vs_to_PSDRs( vs, t_lims )
    space_and_time_description = np.empty( [nrof_pixels, nrof_pixels, sigma_lims.size - 1]  )
    for index in range( sdrs.shape[0] ):
        m_and_n = np.random.randint( low = 0, high = 512, size = 2 )
        space_and_time_description[m_and_n[0], m_and_n[1], :] = psdrs[index, :]
        
    return (space_and_time_description, sigma_lims)


# Return input and output tensors of a tensorflow graph that computes images when given experiment descriptions, for a specific experimental setting
def get_image_creator( discrete_kernels, 
                                  optical_kernel = tf.constant( spatially_integrated_Gaussian( ), dtype = tf.float32 ),
                                  dimension_image = 512, 
                                  nrof_sigma_centers = 30,
                                  quantization_bits_noise = 6 ):
    # Alternatively, load more than one by non-singleton first dimension (discuss with Vidit)
    space_and_time_description_placehoder = tf.placeholder( tf.float32, [ dimension_image,
                                                               dimension_image, 
                                                               nrof_sigma_centers ] )
    # Format, NHWC
    space_and_time_description = tf.expand_dims(  space_and_time_description_placehoder, axis = 0 )
    with tf.name_scope( "forward_operator" ):
        # Format HWCM M: channel multiplier (1)
        discrete_kernels = tf.expand_dims( tf.constant( discrete_kernels, dtype = tf.float32 ), axis = 3 )
        # Convolve each layer
        conv = tf.nn.depthwise_conv2d( input = space_and_time_description,
                                   filter = discrete_kernels,
                                   strides = [1, 1, 1, 1],
                                   padding = 'SAME'  )
        # Add up
        image_pure = tf.expand_dims( tf.reduce_sum( conv, 3 ), axis= 3 )
    with tf.name_scope( "physical_flaws" ):
        # Make an optical blur
        image_without_noise = tf.nn.conv2d( image_pure,
                                                                 tf.expand_dims( tf.expand_dims( optical_kernel, axis = 2 ), axis = 3 ),
                                                                 strides = [1, 1, 1, 1],
                                                                 padding = 'SAME' )
        # Store normalization value and normalize to [0,1]
        normalization = 1/tf.reduce_max( image_without_noise )
        image_without_noise = normalization*image_without_noise
        # Add noise and clip to [0,255]
        standard_deviation = np.sqrt( 2**(-2 * quantization_bits_noise) / 12 )
        image_final = tf.minimum( 255 * tf.maximum( image_without_noise
                          + tf.random_normal( image_without_noise.shape, mean = 0, stddev = standard_deviation ), 0 ), 255 )
    # Normalize input and return
    proportional_space_and_time_description = 255 * normalization * tf.squeeze( space_and_time_description, axis = 0 )
    
    return ( image_final, proportional_space_and_time_description, space_and_time_description_placehoder )


