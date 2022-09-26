#---------------------------------------------------------------------------------------------------#
# File name: helpers.py                                                                             #
# Autor: Chrissi2802                                                                                #
# Created on: 11.09.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Exact description in the functions.
# This file provides auxiliary classes and functions for neural networks.


from datetime import datetime
import tensorflow as tf


class Program_runtime():
    """Class for calculating the programme runtime and outputting it to the console."""

    def __init__(self):
        """Initialisation of the class (constructor). Automatically saves the start time."""

        self.begin()

    def begin(self):
        """This method saves the start time."""

        self.__start = datetime.now()   # start time

    def finish(self, print = True):
        """This method saves the end time and calculates the runtime."""
        # Input:
        # print; boolean, default false, the start time, end time and the runtime should be output to the console
        # Output:
        # self.__runtime; integer, returns the runtime

        self.__end = datetime.now() # end time
        self.__runtime = self.__end - self.__start  # runtime

        if (print == True):
            self.show()

        return self.__runtime

    def show(self):
        """This method outputs start time, end time and the runtime on the console."""

        print()
        print("Start:", self.__start.strftime("%Y-%m-%d %H:%M:%S"))
        print("End:  ", self.__end.strftime("%Y-%m-%d %H:%M:%S"))
        print("Program runtime:", str(self.__runtime).split(".")[0])    # Cut off milliseconds
        print()


def hardware_config(device = "GPU"):
    """This function configures the hardware."""
    # Input:
    # device; string default GPU, which device to use, TPU or GPU
    # Output:
    # strategy; tensorflow MirroredStrategy

    if (device == "TPU"):
        # TPU, use only if TPU is available
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        # GPU, if not available, CPU is automatically selected
        gpus = tf.config.list_logical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy(gpus)

    return strategy


if (__name__ == "__main__"):
    
    # calculating the programme runtime
    Pr = Program_runtime()
    # Code here
    Pr.finish(print = True)

    # configures the hardware
    strategy = hardware_config("GPU")

    with strategy.scope():
        pass
        # Code here

