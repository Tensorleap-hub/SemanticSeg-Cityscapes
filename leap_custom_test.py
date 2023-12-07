import urllib
from os.path import exists
import matplotlib.pyplot as plt
from leap_binder import *
import keras.losses

from cs_sem_seg.data.cs_data import get_cityscapes_data, CATEGORIES_IDS
from cs_sem_seg.metrics import custom_ce_loss



def check_custom_integration():
    print("Testing")
    # res = get_cityscapes_data()
    model_path = 'models/DeeplabV3.h5'
    if not exists(model_path):
        print("Downloading DeeplabV3.h5 for inference")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/example-datasets-47ml982d/domain_gap/DeeplabV3.h5",
            model_path)
    model = tf.keras.models.load_model(model_path)
    responses = subset_images()
    train_res = responses[0]  # [training, validation, test]
    for idx in range(train_res.length):
        image = input_image(idx, train_res)  # get specific image
        mask_gt = ground_truth_mask(idx, train_res)  # get image gt
        gt_vis_res = mask_visualizer(image, mask_gt)
        input_img_tf = tf.convert_to_tensor(np.expand_dims(image, axis=0))
        y_pred = model([input_img_tf])[0] # infer and get model prediction
        y_true_batch = tf.expand_dims(tf.convert_to_tensor(mask_gt), 0)  # convert ground truth bbs to tensor
        y_pred_batch = tf.expand_dims(tf.convert_to_tensor(y_pred), 0)  # convert ground truth bbs to tensor
        new_ls = custom_ce_loss(y_true_batch, y_pred_batch)
        cce_ls = tf.reduce_mean(keras.losses.categorical_crossentropy(y_true_batch, y_pred_batch, from_logits=True, axis=-1))
        bce_ls = tf.reduce_mean(keras.losses.binary_crossentropy(y_true_batch, y_pred_batch, from_logits=True, axis=-1))
        print(f'New Loss: {new_ls}, BCE Loss: {bce_ls}, CCE Loss: {cce_ls}')
        # res = cityscape_segmentation_visualizer(mask_gt)
        # json_data = metadata_json_data(idx, train_res)
        # class_percent = metadata_class(idx, train_res)
        # # np.equal(loss, bi_loss, 0.1)
        # pred_vis_res = mask_visualizer(image, y_pred.numpy())
        # raw_image = non_normalized_input_image(idx, train_res)  # get specific image
        # # visualizers
        # loss_visualizer_res = loss_visualizer(image, y_pred, mask_gt)
        # class_iou_res = class_mean_iou(mask_gt, y_pred)
        # plt.imshow(gt_vis_res.mask)
        # plt.imshow(pred_vis_res.mask)

        # # custom metrics
        # metric_result = mean_iou(y_pred, mask_gt)
        #
        # brightness = metadata_brightness(idx, train_res)
        # filename_city_dataset = metadata_filename_city_dataset(idx, train_res)

if __name__ == "__main__":
    check_custom_integration()