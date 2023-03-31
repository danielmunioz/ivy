# import mxnet
from ivy_tests.test_ivy.test_frontends import NativeClass


mxnet_classes_to_ivy_classes = {}


def convmxnet(argument):
    """Convert NativeClass in argument to ivy frontend counterpart for mxnet"""
    if isinstance(argument, NativeClass):
        return mxnet_classes_to_ivy_classes.get(argument._native_class)
    return argument