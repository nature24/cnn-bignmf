import numpy as np
import tensorflow as tf

from newfunc import nnn
sess = tf.InteractiveSession()

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
matrix1 = tf.constant([[3., 3.]])
print sess.run(matrix1)

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.], [2.]])
print sess.run(matrix2)

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = sess.run(tf.matmul(matrix1, matrix2))

print product

x = tf.ones([2, 3], tf.int32)
print(sess.run(x))

def abc(a):
    print a


abc("hello")
print nnn(10)


test = tf.constant([1, 2, 3])
test2 = tf.constant(-1.0, shape=[2, 3])
print sess.run(test)
print sess.run(test2)

print(sess.run(tf.fill([2,3],2)))
print(sess.run(tf.constant(2,shape=[2,3])))

t=tf.sparse_to_dense([[1,1],[2,1],[2,4],[4,5]],[10,10],[1,2,4,5])
print(sess.run(t))
print(sess.run([test,test2]))
a = tf.Variable([0.0,1.0])
b = tf.placeholder(dtype=tf.float32,shape=[2])
op = tf.assign(a,b)
sess.run(tf.global_variables_initializer())
print(sess.run(a))
sess.run(op,feed_dict={b:[5.,1.]})
print(sess.run(a))
mylist=['sd']
assert len(mylist) >= 1
mylist.pop()
mylist

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])
        assert v.name == "foo/bar/v:0"
