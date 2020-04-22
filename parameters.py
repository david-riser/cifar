import sherpa

def build_sherpa_parameter_space():
    params = [
        sherpa.Ordinal(name='depth', range=[2,3]),
        sherpa.Discrete(name='dense_neurons', range=[64, 256]),
        sherpa.Discrete(name='init_filters', range=[8,64]),
        sherpa.Choice(name='use_batchnorm', range=[False, True]),
        sherpa.Continuous(name='dropout', range=[0.05, 0.55]),
        sherpa.Ordinal(name='batch_size', range=[16, 32, 64, 128, 256, 512, 1024]),
        sherpa.Continuous(name='learning_rate', range=[0.0005, 0.05]),
        sherpa.Continuous(name='beta1', range=[0.5, 1.0]),
        sherpa.Continuous(name='beta2', range=[0.8, 1.0])
    ]
    return params
