import keras.losses
import tensorflow as tf


def get_pixel_loss(gt, pred):

    # Encode gt of multi-label to one hot such that: the matched index = 1 OR only zeros in case of no match.
    max_pred = tf.argmax(pred, -1)  # Prediction: [w, h, c] -> [w, h]
    one_hot_pred = tf.one_hot(max_pred, depth=pred.shape[-1], on_value=1.0, off_value=0.0, axis=-1)  # Prediction: [w, h, 1] -> [w, h, c]
    gt_matched = gt*one_hot_pred      # [w, h, c]
    gt_is_matched = tf.reduce_sum(gt_matched, -1)  # [w, h]     where 1 if there is a match or 0

    # calc loss for pixels with matched prediction: compute CCE when target is the matched one
    match_loss = keras.losses.categorical_crossentropy(gt_matched, pred, from_logits=True, axis=-1)  # [w, h]

    # calc loss for pixels with no matched prediction: compute CCE when dividing the loss in the targets number
    no_match_loss = keras.losses.categorical_crossentropy(gt, pred, from_logits=True, axis=-1)  # [w, h]
    total_labels = tf.cast(tf.reduce_sum(gt, -1), dtype=tf.float32)
    no_match_loss /= tf.where(total_labels > 0, total_labels, 1)  # [w, h] divide the loss in number of total targets

    return tf.where(gt_is_matched > 0, match_loss, no_match_loss)


def custom_ce_loss(y_true, y_pred):
    """
    Calculate Cross Entropy Loss Costumed for Kili segmentation Labels using TensorFlow.

    Args:
        y_true (tf.Tensor): Ground truth segmentation mask tensor - Multi-Label (Supports Occlusion)
        y_pred (tf.Tensor): Predicted segmentation mask tensor - Multi-Class (Without Occlusions).

    Returns:
        tf.Tensor: Loss Values.
    """
    loss = 0.
    batch, n = y_pred.shape[0], y_pred.shape[1]*y_pred.shape[2]     # w*h
    for i in range(batch):
        gt, pred = y_true[i, ...], y_pred[i, ...]
        l_img = get_pixel_loss(gt, pred)    # gt as one hot with 1's in matching index or all 0's
        loss += tf.reduce_mean(l_img)    # average sample loss
    loss /= batch
    return loss
