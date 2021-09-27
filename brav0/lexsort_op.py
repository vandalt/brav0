import numpy as np
import theano
from theano.gradient import grad_undefined
from theano.graph.basic import Apply
from theano.graph.op import Op
from theano.misc.safe_asarray import _asarray
from theano.tensor.basic import mul
from theano.tensor.sort import _variable_is_none


def lexsort(a, axis=-1):
    """
    Returns the indices that would sort arrays
    """
    if axis is None:
        a = a.flatten()
        axis = 0
    # To preserve op shape, it returns extra dim 0 of length one -> take [0]
    output = LexSortOp()(a, axis)

    return output


class LexSortOp(Op):
    """
    This class is a wrapper for numpy lexsort function.
    """

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        axis = theano.tensor.as_tensor_variable(axis)
        bcast = input.type.broadcastable
        return Apply(
            self,
            [input, axis],
            [theano.tensor.TensorType(dtype="int64", broadcastable=bcast)()],
        )

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        if axis is not None:
            if axis != int(axis):
                raise ValueError("sort axis must be an integer or None")
            axis = int(axis)
        z = output_storage[0]
        z[0] = _asarray(
            np.array([np.lexsort(a, axis=axis)]),
            dtype=node.outputs[0].dtype,
        )

    def infer_shape(self, fgraph, node, inputs_shapes):
        if _variable_is_none(node.inputs[1]):
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None, so there should be the same number of
        # dimensions in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        # No grad defined for intergers.
        inp, axis = inputs
        inp_grad = inp.zeros_like()
        axis_grad = grad_undefined(
            self,
            1,
            axis,
            "lexsort is not defined for non-integer axes so"
            " lexsort(x, axis+eps) is undefined",
        )
        return [inp_grad, axis_grad]

    """
    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    """
