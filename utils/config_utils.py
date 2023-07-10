import yaml

def load_config(path): 
    with open(path, "r", encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    return data

class DictToClass(object):
    @classmethod
    def _to_class(cls, _obj):
        _obj_ = type('cfg', (object,), _obj)
        [setattr(_obj_, key, cls._to_class(value)) if isinstance(value, dict) else setattr(_obj_, key, value) for
         key, value in _obj.items()]
        return _obj_

if __name__ =='__main__':
    config = load_config('config\LVP_TuSimple.yaml')
    print(config)
    config = DictToClass._to_class(config)