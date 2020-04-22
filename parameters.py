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
