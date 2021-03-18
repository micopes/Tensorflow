# Mutable(constant) and Immutable(Variable)

<hr>

### list와 np.ndarray

`test_list = [1, 2, 3]`
- list
<br>

`test_np = np.array([1, 2, 3])`
- np.ndarray
<br>
로 test_list와 test_np는 다른 타입이지만,

  - tf.constant(), tf.Variable()을 적용하면 각각 EagarTensor와 ResourceVariable로 타입이 같아진다.

<hr>

### tf.constant()와 tf.Variable()

tf.constant()
- Mutable
- type: `<class 'tensorflow.python.framework.ops.EagerTensor'>`

tf.Variable()
- Immutable.
- type: `<class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>`

t1 = tf.constant(test_list)
t2 = tf.constant(test_list)

<hr>

### tf.constant와 tf.Variable의 타입 변환

`t1 = tf.constant(test_list)`

- `tf.Variable(t1)` -> **Error!**

`t2 = tf.Variable(test_list)`

- `tf.constant(t2)` -> Possible!

<br>
tf.constant() -> tf.Variable()로 바꿔주려면

>tf.constant() 대신에 동일한 타입(EagarTensor)이지만, 

>변경가능한 **tf.convert_to_tensor()**로 원래의 텐서를 사용하고 tf.Variable()로 바꿔줘야 한다.

<hr>

### 더하면 타입이 어떻게 변할까?

- `tf.constant() + tf.constant() -> EagarTensor(Immutable)`
<br>
- `tf.constant() + tf.Variable() -> EagarTensor(Immutable)`
<br>
- `tf.Variable() + tf.Variable() -> EagarTensor(Immutable)`

