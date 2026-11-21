"""Commit-pinned TACO checker in the same CPU sandbox as ordinary benchmarks."""
import json
import multiprocessing as mp
from pathlib import Path
import sys


def _child(root,sample,solution,connection):
    try:
        import inspect
        from collections import namedtuple
        # Upstream pyext2 predates Python3.11's removal of getargspec.
        # Restore the old four-field signature, without changing test comparison.
        if not hasattr(inspect,'getargspec'):
            spec_type=namedtuple('ArgSpec','args varargs keywords defaults')
            def argspec(function):
                v=inspect.getfullargspec(function)
                return spec_type(v.args,v.varargs,v.varkw,v.defaults)
            inspect.getargspec=argspec
        sys.path.insert(0,str(root))
        from metrics.testing_util import run_test
        values=run_test(sample,test=solution,debug=False)
        connection.send({'tests':[int(v) for v in values]})
    except BaseException as exc:
        connection.send({'infrastructure_error':type(exc).__name__+': '+str(exc)})
    finally:connection.close()


def check(task,solution,root):
    sample={'input_output':task['raw']['input_output']}
    if not isinstance(sample['input_output'],str):sample['input_output']=json.dumps(sample['input_output'])
    tests=json.loads(sample['input_output'])
    if not tests.get('inputs') or len(tests['inputs'])!=len(tests.get('outputs',[])):
        raise ValueError('Invalid TACO tests')
    if not (Path(root)/'metrics/testing_util.py').is_file():raise ValueError('Pinned TACO checker missing')
    parent,child=mp.Pipe(duplex=False)
    process=mp.Process(target=_child,args=(root,sample,solution,child));process.start();child.close()
    wall_limit=min(600,max(30,5*len(tests['inputs'])+10))
    if not parent.poll(wall_limit):
        process.kill();process.join();parent.close()
        return {'correct':False,'tests':[],'stop_reason':'outer_timeout','wall_limit_seconds':wall_limit}
    result=parent.recv();parent.close();process.join(timeout=5)
    if process.is_alive():process.kill();process.join()
    if 'infrastructure_error' in result:raise RuntimeError(result['infrastructure_error'])
    return {'correct':bool(result['tests']) and all(v==1 for v in result['tests']),
            'tests':result['tests'],'stop_reason':'completed','wall_limit_seconds':wall_limit}
