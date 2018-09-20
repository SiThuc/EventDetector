import io
import ruamel
import ruamel.yaml as ya

class Config(object):

    def load(paraFile):
        # Read YAML file
        with open(paraFile, 'r') as stream:
            config = ya.load(stream, Loader= ruamel.yaml.Loader)
        return config