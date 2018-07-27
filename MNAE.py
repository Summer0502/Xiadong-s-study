import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# from tensorflow.contrib.factorization.examples.mnist import fill_feed_dict
# 用Denosing AutoEncoder对minist进行重构实验
def xavier_init(fan_in, fan_out, const=1):
    # Xavier法对各个权重进行初始化，比较适合各种激活函数
    low = -const * np.sqrt(6.0 / (fan_in + fan_out))
    high = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low, maxval=high,
                             dtype=tf.float32)


class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()

        self.weights = network_weights

        self.x = tf.placeholder(tf.float32, [None, self.n_input])  # 输入层
        self.n_hidden = self.transfer(tf.add(tf.matmul(  # 隐含层，输入加上噪声乘以w加b
            self.x + scale * tf.random_normal((n_input,)),
            self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.n_hidden,
                                               self.weights['w2']), self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)  # 优化器相当方便，只要输入一个cost函数就可以

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):  # 权重生成器
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros(self.n_hidden,
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        # fit方法一般接受cost和优化器，还有scale等训练的时候要用的东西，并开始run
        # fit方法对输入的X进行训练子图，一般是传入一个batch进行训练
        cost, opt = self.sess.run((self.cost, self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        # cost方法根据X进行计算cost，是验证的时候计算总cost的时候用的
        return self.sess.run(self.cost, feed_dict={self.x: X,
                                                   self.scale: self.training_scale
                                                   })

    def transform(self, X):
        # 中间层的接口，跟如传入的X计算中间层输出，注意，hidden是第一层输出乘第二层权重，
        # 也就是说返回的是前两层的结果
        # 因此可以说是获取中间层输出的子图（不是训练子图）
        return self.sess.run(self.hidden, feed_dict={self.x: X,
                                                     self.scale: self.training_scale
                                                     })

    def generate(self, hidden=None):
        # 还原子图，以后这种子图在模型训练好之后，具有单独的特征提取和特征还原功能
        if hidden is None:
            hidden = np.random.normal(size=self.weight['b1'])
        return self.sess.run(self.reconstruction,
                             feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        # 单独定义的还原子图，本质上是上面那个函数的子函数
        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                             self.scale: self.training_scale
                                                             })

    def getWeights(self):
        # 获取中间层的权重
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])


mnist = input_data.read_data_sets("E:/untitled2018MNAE/MNIST_data", one_hot=True)


def standard_scale(X_train, X_test):
    # 分别进行standard操作
    preprocess = prep.StandardScaler().fit(X_train)
    # 在训练数据上fit的scaler，在test上也可以用。
    # 并且为什么不一起standard呢？因为考虑训练数据在训练的时候要有均值假设，不能一扔进去就不是均值的了
    X_train = preprocess.transform(X_train)
    X_test = preprocess.transform(X_test)
    return X_train, X_test


def get_random_block_from_data(data, batch_size):
    # 随机抽取block，不放回抽样，提高数据的使用率
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]


X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 200
batch_size = 128
display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                               n_hidden=256,
                                               transfer_function=tf.nn.softplus,
                                               # 函数可以加括号也可以不加括号，不加括号就不用参数，自动适应
                                               optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                               scale=0.01)
# 每一个batch中，获取一个block，进行fit，对cost做平均每个样本的cost，累加
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost = ",
              "{:.9f}".format(avg_cost))

print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))
