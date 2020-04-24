import sherpa

def build_sherpa_parameter_space():
    params = [
        sherpa.Ordinal(name='depth', range=[2,3]),
        sherpa.Discrete(name='dense_neurons', range=[100, 164]),
        sherpa.Discrete(name='init_filters', range=[8,32]),
        sherpa.Choice(name='use_batchnorm', range=[False, True]),
        sherpa.Continuous(name='dropout', range=[0.35, 0.55]),
        sherpa.Ordinal(name='batch_size', range=[512, 1024]),
        sherpa.Continuous(name='learning_rate', range=[0.005, 0.01]),
        sherpa.Continuous(name='beta1', range=[0.45, 0.55]),
        sherpa.Continuous(name='beta2', range=[0.95, 1.0])
    ]
    return params

def build_sherpa_augmentations_space():
    params = [
        sherpa.Continuous(name='width_shift', range=[0.0, 0.2]),
        sherpa.Continuous(name='height_shift', range=[0.0, 0.2]),
        sherpa.Continuous(name='zoom', range=[0.0, 0.3]),
        sherpa.Choice(name='horizontal_flip', range=[False, True]),
        sherpa.Discrete(name='rotation', range=[0, 30])
    ]
    return params
