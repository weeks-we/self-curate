"""Two disjoint four-GPU lanes, asynchronous CPU grading, atomic shared state."""
import copy
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import RLock,Event
from pathlib import Path
from .common import write,sha,event


class Pipeline:
    def __init__(self,backend,config,output,state,state_file,section):
        self.backend=backend;self.config=config;self.output=Path(output);self.state=state
        self.state_file=state_file;self.section=section;self.lock=RLock();self.abort=Event();self.errors=[];self.futures=[]
        spec=config['runtime']['pipeline'];self.spec=spec
        self.graders=ThreadPoolExecutor(max_workers=spec['cpu_grading_workers'])
        self.lanes=[]
        ids=config['hardware']['gpu_uuids']
        if backend.kind=='real' and (len(ids)!=8 or len(set(ids))!=8):raise ValueError('Pipeline needs eight exclusive authorized GPUs')
        for i in range(2):
            lane=backend.fork_lane(i) if hasattr(backend,'fork_lane') else copy.copy(backend)
            lane.port_base=spec['base_port']+i*spec['port_stride'];lane.cancel_event=self.abort
            cfg=copy.deepcopy(config);cfg['hardware']['gpu_uuids']=ids[i*4:(i+1)*4]
            cfg['hardware']['gpu_count']=4;cfg['evaluation']['replicas']=4
            self.lanes.append((lane,cfg))
        write(self.output/'pipeline-layout.json',{'synthetic':backend.kind=='mock','lanes':[
            {'lane':i,'gpu_uuids':c['hardware']['gpu_uuids'],'world_size':c['training']['world_size'],
             'global_batch_size':c['training']['global_batch_size'],'base_port':b.port_base} for i,(b,c) in enumerate(self.lanes)],
            'cpu_grading_workers':spec['cpu_grading_workers'],'assignment':'stable job index modulo2'})

    def record(self,key):
        with self.lock:return copy.deepcopy(self.state.get(self.section,{}).get(key,{}))

    def put(self,key,phase,**values):
        with self.lock:
            entry=self.state.setdefault(self.section,{}).setdefault(key,{})
            entry.update(copy.deepcopy(values));self.state.setdefault('in_flight',{})[key]=phase
            write(self.state_file,self.state)
            event(self.output,'pipeline_progress',job=key,phase=phase)
            print('[pipeline] '+key+' '+phase,flush=True)

    def evaluation(self,key,label,value):
        with self.lock:
            self.state[self.section][key].setdefault('evaluations',{})[label]=value
            write(self.state_file,self.state)

    def fail(self,exc):
        with self.lock:
            if not self.errors:self.errors.append(exc)
        self.abort.set()
        for lane,_ in self.lanes:
            if hasattr(lane,'stop'):lane.stop()

    def submit(self,finish,key,label):
        def task():
            if self.abort.is_set():return
            try:
                value=finish();self.evaluation(key,label,value)
            except BaseException as exc:self.fail(exc);raise
        self.futures.append(self.graders.submit(task))

    def evaluate(self,lane,cfg,job,model,benchmark,folder,key,label,resume,tasks=None):
        cached=self.record(key).get('evaluations',{}).get(label)
        if cached:
            lane.verify_result(cached);return
        if self.abort.is_set():raise RuntimeError('Pipeline aborted')
        self.put(key,'generate:'+label)
        if self.backend.kind=='mock':
            self.submit(lambda:lane.evaluate(job,cfg,model,benchmark,folder,resume=resume),key,label)
        elif tasks is not None:
            from .nice_sandbox import prepare_tasks,finish_tasks
            ticket=prepare_tasks(lane,cfg,model,tasks,Path(folder)/'taco_dev')
            self.submit(lambda:finish_tasks(ticket),key,label)
        else:
            ticket=lane.prepare_evaluation(job,cfg,model,benchmark,folder,resume=resume)
            self.submit(lambda:lane.finish_evaluation(ticket),key,label)

    def run(self,jobs,callback):
        def worker(index):
            lane,cfg=self.lanes[index]
            try:
                for position in range(index,len(jobs),2):
                    if self.abort.is_set():return
                    callback(position,jobs[position],lane,copy.deepcopy(cfg),self)
            except BaseException as exc:self.fail(exc)
            finally:
                if hasattr(lane,'stop'):lane.stop()
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(worker,i) for i in range(2)]
                for f in as_completed(futures):f.result()
            for f in self.futures:
                try:f.result()
                except BaseException as exc:self.fail(exc)
        finally:
            self.graders.shutdown(wait=True,cancel_futures=self.abort.is_set())
        if self.errors:raise self.errors[0]
        with self.lock:
            self.state['in_flight']={};self.state['pending']=None;write(self.state_file,self.state)


def ordinary_jobs(value,backend,output,state,state_file,resume):
    def run(index,job,lane,cfg,pipe):
        if sha(job['data'])!=job['data_sha256']:raise ValueError('Selected data changed')
        key=job['job_id'];folder=Path(output)/key;folder.mkdir(exist_ok=True)
        if job.get('learning_rate') is not None:cfg['training']['learning_rate']=job['learning_rate']
        prior=pipe.record(key)
        if 'model' not in prior:
            pipe.put(key,'train')
            model=lane.initial(cfg) if job['method']=='initial' else lane.train(job,cfg,folder,resume=resume)
            pipe.put(key,'model_published',model=model)
        else:model=prior['model']
        if hasattr(lane,'verify_model'):lane.verify_model(model)
        if job['method']=='initial':stages=[('initial',model)]
        else:
            saved=model['epoch_checkpoints']
            if [r['epoch'] for r in saved]!=[1,2,3]:raise ValueError('All epoch receipts required')
            stages=[('epoch-'+str(s['epoch']).zfill(2),s['model']) for s in saved if s['epoch'] in cfg['evaluation']['epochs']]
        for name,stage in stages:
            if hasattr(lane,'verify_model'):lane.verify_model(stage)
            stage_folder=folder/name;stage_folder.mkdir(exist_ok=True)
            for benchmark in job['benchmarks']:
                pipe.evaluate(lane,cfg,job,stage,benchmark,stage_folder,key,name+'/'+benchmark,resume)
    Pipeline(backend,value['config'],output,state,state_file,'completed').run(value['jobs'],run)


def nice_jobs(plan,base,backend,root,state,state_file,resume):
    from .epoch_checkpoints import progress_steps
    jobs=[(group,selection,lr) for group,selection in state['selections'].items() for lr in plan['learning_rates']]
    steps=progress_steps(1000,base['training']['global_batch_size'],base['training']['save_at_fractions'])
    def run(index,descriptor,lane,cfg,pipe):
        group,selection,lr=descriptor
        if sha(selection['data'])!=selection['data_sha256']:raise ValueError('Frozen NICE subset changed')
        key=group+'-lr'+format(lr,'.8g');cfg['training']['learning_rate']=lr
        job={**selection,'job_id':'nice-'+key,'domain':'coding','method':group,'benchmarks':plan['nice']['evaluation']['benchmarks']}
        folder=Path(root)/'branches'/key;folder.mkdir(parents=True,exist_ok=True)
        prior=pipe.record(key);pipe.put(key,'train',group=group,learning_rate=lr)
        if 'model' not in prior:
            model=lane.train(job,cfg,folder,resume=resume);pipe.put(key,'model_published',model=model)
        else:model=prior['model']
        stages=model['epoch_checkpoints']
        if [v['global_step'] for v in stages]!=steps:raise ValueError('Wrong NICE progress checkpoints')
        for i,stage in enumerate(stages,1):
            name='progress-'+str(i);model=stage['model']
            if hasattr(lane,'verify_model'):lane.verify_model(model)
            stage_folder=folder/name;stage_folder.mkdir(exist_ok=True)
            for benchmark in ['taco_dev']+job['benchmarks']:
                pipe.evaluate(lane,cfg,job,model,benchmark,stage_folder,key,name+'/'+benchmark,resume,
                    tasks=plan['tasks'] if benchmark=='taco_dev' else None)
    Pipeline(backend,base,root,state,state_file,'branches').run(jobs,run)
