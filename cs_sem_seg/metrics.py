import keras.losses
import tensorflow as tf

from cs_sem_seg.data.cs_data import CATEGORIES


def class_mean_iou(y_true, y_pred) -> dict:
    """
    Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

    Args:
        y_true (tf.Tensor): Ground truth segmentation mask tensor.
        y_pred (tf.Tensor): Predicted segmentation mask tensor.

    Returns:
        tf.Tensor: Mean Intersection over Union (mIOU) value.
    """
    res = {}
    for i, c in enumerate(CATEGORIES):
        y_true_i, y_pred_i = y_true[..., i], y_pred[..., i]
        res[f'{c}'] = mean_iou(y_true_i, y_pred_i)
    return res


def get_class_mean_iou(class_i: int = None):

    def class_mean_iou(y_true, y_pred):
        """
        Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

        Args:
            y_true (tf.Tensor): Ground truth segmentation mask tensor.
            y_pred (tf.Tensor): Predicted segmentation mask tensor.

        Returns:
            tf.Tensor: Mean Intersection over Union (mIOU) value.
        """
        y_true, y_pred = y_true[..., class_i], y_pred[..., class_i]
        iou = mean_iou(y_true, y_pred)

        return iou

    return class_mean_iou


def mean_iou(y_true, y_pred):
    """
    Calculate the mean Intersection over Union (mIOU) for segmentation using TensorFlow.

    Args:
        y_true (tf.Tensor): Ground truth segmentation mask tensor.
        y_pred (tf.Tensor): Predicted segmentation mask tensor.

    Returns:
        tf.Tensor: Mean Intersection over Union (mIOU) value.
    """
    # Flatten the tensors
    y_true_flat = tf.reshape(y_true, [y_true.shape[0], -1])
    y_pred_flat = tf.cast(tf.reshape(y_pred, [y_true.shape[0], -1]), y_true_flat.dtype)

    # Calculate the intersection and union
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, -1)
    union = tf.reduce_sum(tf.maximum(y_true_flat, y_pred_flat), -1)

    # Calculate the IOU value
    iou = tf.where(union > 0, intersection / union, 0)

    return iou


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
        # TODO FLatten
        max_pred = tf.argmax(pred, -1)     # Prediction: [w, h, c] -> [w, h, 1]
        one_hot_pred = tf.cast(tf.one_hot(max_pred, depth=pred.shape[-1], on_value=1.0, off_value=0.0, axis=-1), dtype=tf.int32)      # Prediction: [w, h, 1] -> [w, h, c]
        gt_matched = tf.multiply(gt, one_hot_pred, -1)   # [w, h, c]
        gt_sum_matched = tf.reduce_sum(gt_matched, -1)   # [w, h]
        match_loss = keras.losses.categorical_crossentropy(gt_matched, pred, from_logits=True, axis=-1)#, reduce='none')        # TODO: check that what the model outputs
        no_match_loss = keras.losses.categorical_crossentropy(gt, pred, from_logits=True, axis=-1)
        no_match_loss /= tf.cast(tf.reduce_sum(gt, -1), dtype=tf.float32)    # [w, h] divide the loss in number of actual targets
        match_loss = tf.reduce_sum(tf.boolean_mask(match_loss, gt_sum_matched > 0))
        no_match_loss = tf.reduce_sum(tf.boolean_mask(no_match_loss, gt_sum_matched == 0))
        loss += ((match_loss + no_match_loss) / n)    # average sample loss
    loss /= batch
    return loss
















