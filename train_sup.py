import importlib
from pprint import pprint
from trainer.suptrainer import SupParams
import trainer
from pkgutil import iter_modules

methods = {i.name for i in list(iter_modules(trainer.__path__))}

if __name__ == '__main__':
    params = SupParams()
    params.module = None
    params.from_args()
    if params.module is None or params.module not in methods:
        print('--module=')
        pprint(methods)
        exit(1)
    module = importlib.import_module(f'trainer.{params.module}')
    module.main()
