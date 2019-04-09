from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import SpatialDropout1D
from keras.layers import concatenate
from keras.models import Model
import numpy as np


def NN_with_embedding(
    data: DataFrame,
    cat_vars: StrList,
    contin_vars: StrList,
    units: Collection[int] = [16],
    drop_fraction: Collection[float] = None,
    n_classes: int = 0,
):
    """ Fully connected NN with category embedding

    The model accepts as input a combination of categorical and continuous inputs.
    it can be used for
    * regression (`n_classes=0`)
    * binary classification (`n_classes=1`)
    * multiclass classification (`n_classes>1`)


    :param data: input data, or a reasonable representation of it
    :type data: DataFrame
    :param cat_vars: List of categorical variables
    :type cat_vars: StrList
    :param contin_vars: List of continuous variables
    :type contin_vars: StrList
    :param units: number of neurons for each layer
    :type units: Collection[int]=[16]
    :param drop_fraction: fraction of DropOut for each layer, defaults to None
    :type drop_fraction: Collection[float], optional
    :param n_classes: Output classes, defaults to 0
    :type n_classes: int, optional
    :returns: model
    :rtype: Keras model
    """

    # loop over categorical variables to prepare Input and Embedding layers
    cat_inputs = []
    emb_layers = []
    for c in cat_vars:
        cat_sz = data[c].nunique() + 1
        temp_input = Input(shape=[1], name=c)
        temp_emb = Embedding(
            cat_sz,
            # min(50, (cat_sz + 1) // 2),
            min(600, round(1.6 * cat_sz ** 0.56)),
            name=c + "_embedding",
        )(temp_input)
        cat_inputs.append(temp_input)
        emb_layers.append(temp_emb)

    # Concatenate Embedded vectors
    cat_concat = concatenate(emb_layers)
    s_dout = SpatialDropout1D(0.0001)(cat_concat)

    # Concatenate Inputs and Embedded layers
    cont_inputs = [Input(shape=[1], name=c) for c in contin_vars]
    x = concatenate([Flatten()(s_dout), *cont_inputs])

    # prepares the parameter for the fully connected layers
    if drop_fraction is None:
        drop_fraction = np.zeros_like(units)
    if not isinstance(drop_fraction, list):
        drop_fraction = [drop_fraction] * len(units)

    # Deep layers
    for i, (n, d) in enumerate(zip(units, drop_fraction)):

        x = Dense(n, activation="relu", name=f"Dense layer {i}")(x)
        x = BatchNormalization(name=f"BN {i}")(x)
        x = Dropout(d, name=f"Dropput {i}")(x)

    # output
    if n_classes == 0:
        x = Dense(1)(x)
    elif n_classes == 1:
        x = Dense(1, activation="sigmoid")(x)
    else:
        x = Dense(n_classes, activation="softmax")(x)

    # resulting model
    model = Model(inputs=cat_inputs + cont_inputs, outputs=x)

    return model
