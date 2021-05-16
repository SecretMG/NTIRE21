import tensorflow as tf
import keras
import numpy as np
import matplotlib.cm as cm

def create_cam(input, model, last_conv_layer_name, classifier_layer_names):
    '''
    Create class activation map(CAM) base on your model and the input
    :param input: the original figure, needing arrary type
    :param model: the classifier model
    :param last_conv_layer_name: name of the last convlution layer of model
    :param classifier_layer_names: name list after the last convlution layer of model
    :return: the CAM
    '''
    X = keras.applications.xception.preprocess_input(input)
    X = np.expand_dims(X, axis=0)

    # 重建卷积模型
    last_conv_layer = model.get_layer(last_conv_layer_name)
    conv_model = keras.Model(model.inputs, last_conv_layer.output)
    # conv_model.summary()

    # 重建检测模型
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)
    # classifier_model.summary()

    # 求预测向量最大值对最后一层卷积特征图的梯度
    with tf.GradientTape() as tape:
        last_conv_layer_output = conv_model(X)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    print(grads.shape)

    # 取全局平均
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    # 特征图通道维度上加权平均计算热力图
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    # plt.matshow(heatmap)
    # plt.show()

    # 将热力图叠加到原图上
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap('jet')
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((input.shape[1], input.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + input*255
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img