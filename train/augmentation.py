import tensorflow as tf

def add_noise(sequence, noise_stddev_range=(0.005, 0.02)):
    """Adds random noise to the keypoint coordinates."""
    noise_stddev = tf.random.uniform([], minval=noise_stddev_range[0], maxval=noise_stddev_range[1])
    noise = tf.random.normal(shape=tf.shape(sequence), stddev=noise_stddev)
    return sequence + noise

def random_scale(sequence, scale_range=(0.85, 1.15)):
    """Randomly scales keypoint coordinates (simulates distance variation)."""
    scale_factor = tf.random.uniform(shape=[], minval=scale_range[0], maxval=scale_range[1])
    return sequence * scale_factor

def temporal_shift(sequence, max_shift_fraction=0.1):
    """Randomly shifts the sequence frames forward or backward."""
    seq_len = tf.shape(sequence)[0]
    max_shift = tf.cast(tf.cast(seq_len, tf.float32) * max_shift_fraction, tf.int32)
    max_shift = tf.minimum(max_shift, seq_len // 2)
    shift_amount = tf.random.uniform(shape=[], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32)
    return tf.roll(sequence, shift=shift_amount, axis=0)

# Combine augmentations probabilistically
def apply_augmentations(sequence, label):
    """Applies a set of augmentations probabilistically to a sequence-label pair."""
    sequence = tf.cast(sequence, dtype=tf.float32)
    # Apply augmentations based on probability
    if tf.random.uniform([]) < 0.7: # 70% chance to add noise
        sequence = add_noise(sequence)
    if tf.random.uniform([]) < 0.5: # 50% chance to scale
        sequence = random_scale(sequence)
    if tf.random.uniform([]) < 0.5: # 50% chance to temporal shift
         sequence = temporal_shift(sequence)
    return sequence, label