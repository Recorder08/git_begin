import tensorflow as tf  # 완전히 Python 밖에서 실행되는 서로 상호작용하는 작업들의 그래프를 기술하도록 합니다.
from tensorflow.examples.tutorials.mnist import input_data # input_data를 임폴트합니다.

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784]) #수식기호
W = tf.Variable(tf.zeros([784, 10])) # tensorflow의 상호작용하는 작업 그래프간에 유지되는 변경가능한 텐서
b = tf.Variable(tf.zeros([10]))      # tensorflow의 상호작용하는 작업 그래프간에 유지되는 변경가능한 텐서
y = tf.nn.softmax(tf.matmul(x, W) + b) # x와 W를 곱하고 b를 도하고 마지막으로 tf.nn.softmax를 적용한다.

y_ = tf.placeholder(tf.float32, [None, 10]) #교차 엔트로피를 구하기 위해 새 placeholder를 추가

cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # y의 각 원소의 로그값 계산 후 y_의 각 원소에, 각각에 해당되는 tf.log(y)를 곱합니다. 마지막으로,tf.reduce_sum은 텐서의 모든 원소를 더합니다.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #tensorflow에게 학습도를 0.01로 경사준 하강법 알고리즘을 적용하여 교차엔트로피를 최소화 하도록 명령.

# Session
init = tf.initialize_all_variables() # 세션에서 모델을 시작하고 변수들을 초기화하는 작업을 실행할 수 있습니다.

sess = tf.Session()
sess.run(init)

# Learning
for i in range(1000): # 학습을 1000번 시킴!
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}) #각 반복 단계마다, 학습 세트로부터 100개의 무작위 데이터들의 일괄 처리(batch)들을 가져옵니다. placeholders를 대체하기 위한 일괄 처리 데이터에 train_step 피딩을 실행합니다.

# Validation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 부울리스트를 준다.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 테스트의 정확도 확인

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})) # 정확도를 프린트한다!
