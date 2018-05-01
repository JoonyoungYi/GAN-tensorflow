import tensorflow as tf

from configs import *


def generator(Z, G_Ws, G_bs):
    layer = Z
    for G_W, G_b in zip(G_Ws[:-1], G_bs[:-1]):
        layer = tf.nn.relu(tf.matmul(layer, G_W) + G_b)
    return tf.nn.sigmoid(tf.matmul(layer, G_Ws[-1]) + G_bs[-1])


def discriminator(X, D_Ws, D_bs):
    layer = X
    for D_W, D_b in zip(D_Ws[:-1], D_bs[:-1]):
        layer = tf.nn.relu(tf.matmul(layer, D_W) + D_b)
    return tf.nn.sigmoid(tf.matmul(layer, D_Ws[-1]) + D_bs[-1])


def init_models():
    X = tf.placeholder(tf.float32, [None, D_INPUT_LAYER_NODE_NUMBER])
    Z = tf.placeholder(tf.float32, [None, G_INPUT_LAYER_NODE_NUMBER])

    D_Ws, D_bs = [], []
    for layer_idx in range(D_HIDDEN_LAYER_NUMBER):
        if layer_idx == 0:
            i_node_number = D_INPUT_LAYER_NODE_NUMBER
            o_node_number = HIDDEN_LAYER_NODE_NUMBER
        elif layer_idx == D_HIDDEN_LAYER_NUMBER - 1:
            i_node_number = HIDDEN_LAYER_NODE_NUMBER
            o_node_number = 1
        else:
            i_node_number = HIDDEN_LAYER_NODE_NUMBER
            o_node_number = HIDDEN_LAYER_NODE_NUMBER
        D_Ws.append(
            tf.Variable(
                tf.random_normal([i_node_number, o_node_number], stddev=0.01)))
        D_bs.append(tf.Variable(tf.zeros([o_node_number])))

    G_Ws, G_bs = [], []
    for layer_idx in range(G_HIDDEN_LAYER_NUMBER):
        if layer_idx == 0:
            i_node_number = G_INPUT_LAYER_NODE_NUMBER
            o_node_number = HIDDEN_LAYER_NODE_NUMBER
        elif layer_idx == G_HIDDEN_LAYER_NUMBER - 1:
            i_node_number = HIDDEN_LAYER_NODE_NUMBER
            o_node_number = D_INPUT_LAYER_NODE_NUMBER
        else:
            i_node_number = HIDDEN_LAYER_NODE_NUMBER
            o_node_number = HIDDEN_LAYER_NODE_NUMBER
        G_Ws.append(
            tf.Variable(
                tf.random_normal([i_node_number, o_node_number], stddev=0.01)))
        G_bs.append(tf.Variable(tf.zeros([o_node_number])))

    G = generator(Z, G_Ws, G_bs)
    D_fake = discriminator(G, D_Ws, D_bs)
    D_real = discriminator(X, D_Ws, D_bs)

    # D_E: Expectation of discriminator
    # G_E: Expectation of generator
    D_E = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_fake))
    G_E = tf.reduce_mean(tf.log(D_fake))

    D_train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
        -D_E, var_list=D_Ws + D_bs)
    G_train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
        -G_E, var_list=G_Ws + G_bs)
    return {
        'X': X,
        'Z': Z,
        'G': G,
        'D_train': D_train,
        'D_E': D_E,
        'G_train': G_train,
        'G_E': G_E
    }
